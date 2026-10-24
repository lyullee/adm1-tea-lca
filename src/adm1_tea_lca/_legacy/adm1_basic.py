import numpy as np
from scipy.integrate import solve_ivp
from decimal import Decimal, getcontext


getcontext().prec = 50  # 소수점 50자리 정밀도


class ADM1:
    def __init__(self, v_liq, v_gas, time_range, input_data, initial_values, parameters):
        # 액상 및 기체 구간의 부피를 Decimal로 저장하여 계산의 정밀성을 높임
        self.V_liq = Decimal(v_liq)  # 액상 부피
        self.V_gas = Decimal(v_gas)  # 기체 부피
        # Weinrich public Model_data uses 1.01325 bar. The former 1.0313
        # literal was a transposition error and changed gas-outlet results.
        self.p_atm = Decimal("1.01325")

        # 외부에서 주어진 입력 데이터를 저장
        self.input_data = input_data  # 모델에 필요한 입력 데이터
        self.para = parameters  # 모델의 파라미터 집합
        self.time_range = time_range  # 시뮬레이션 시간 범위 (시작 시간, 종료 시간)
        self.initial_values = initial_values  # 초기 상태 변수 값

    def solve_adm1(self):
        """
        ADM1 모델을 ODE 시스템으로 정의하고, 이를 수치적으로 풀어 모델의 동적 거동을 계산

        Returns:
            t_output (np.ndarray): 시간 값 배열
            y_output (np.ndarray): 각 시간에서의 상태 변수 값 배열
        """

        # ODE 시스템 정의
        def ode_system(t, x):
            """
            ODE 시스템의 정의
            Args:
                t (float): 현재 시간
                x (list): 현재 상태 변수 벡터

            Returns:
                list: 상태 변수의 시간에 대한 변화율 (dx/dt)
            """
            return self.adm1_mass(t, x)

        # 초기 조건을 설정 (초기 상태 변수를 모델 초기 값으로 변환)
        initial_condition = self.get_x_from_initial(self.initial_values)

        # solve_ivp 함수로 ODE 시스템을 풀이 (BDF 방법 사용)
        # BDF(Backward Differentiation Formula)는 강하게 강성(Stiff)한 시스템에 적합
        solution = solve_ivp(
            ode_system,  # ODE 시스템 정의
            self.time_range,  # 시뮬레이션 시간 범위 (t0, tf)
            initial_condition,  # 초기 조건 배열
            method='BDF',  # 강성 시스템에 적합한 수치적 풀이법
            rtol=1.0e-3,
            atol=1.0e-6,
            max_step=1.0,
        )

        # 결과를 저장할 배열 생성
        t_output = np.zeros(len(solution.t))  # 시간 출력 배열 초기화
        y_output = np.zeros((len(solution.t), 44))  # 상태 변수 출력 배열 초기화 (44개의 변수)

        # 시간과 상태 변수 결과를 배열에 저장
        for i in range(len(solution.t)):
            t_output[i] = solution.t[i]  # 시간 값을 저장
            y_output[i, :] = self.adm1_mass_output(solution.y[:, i])  # 상태 변수 값을 저장

        # 시간 값과 상태 변수 값을 반환
        return t_output, y_output

    def adm1_mass_output(self, x):
        """
        ADM1 모델에 기반하여 질량 출력값을 계산하는 함수.

        Args:
            x (list 또는 np.ndarray): 현재 상태 변수.

        Returns:
            np.ndarray: 계산된 출력값을 포함하는 44개의 요소로 구성된 배열.
        """
        # 액상에서 암모늄 농도(S_nh4_i)를 계산
        S_nh4_i = Decimal(x[10]) - Decimal(x[30])
        # 액상에서 이산화탄소 농도(S_co2)를 계산
        S_co2 = Decimal(x[9]) - Decimal(x[29])
        # pH 균형에 관련된 대수 변수 phi를 계산
        phi = Decimal(x[23]) + S_nh4_i / Decimal(17) \
              - Decimal(x[29]) / Decimal(44) - Decimal(x[28]) / Decimal(60) \
              - Decimal(x[27]) / Decimal(74) - Decimal(x[26]) / Decimal(88) \
              - Decimal(x[25]) / Decimal(102) - Decimal(x[24])
        # phi 값과 물의 이온화 상수(K_w)를 사용하여 수소 이온 농도(S_H)를 계산
        S_H = -Decimal(phi) * Decimal(0.5) + Decimal(0.5) * Decimal(phi * phi + 4 * Decimal(self.para.K_w)).sqrt()
        # 수소 이온 농도를 사용하여 pH 값을 계산
        pH = -Decimal(S_H).log10()
        # 수소 가스의 부분 압력(p_h2)을 계산
        p_h2 = Decimal(x[31]) * Decimal(self.para.R) * Decimal(self.para.T) / Decimal(2)
        # 메탄 가스의 부분 압력(p_ch4)을 계산
        p_ch4 = Decimal(x[32]) * Decimal(self.para.R) * Decimal(self.para.T) / Decimal(16)
        # 이산화탄소 가스의 부분 압력(p_co2)을 계산
        p_co2 = Decimal(x[33]) * Decimal(self.para.R) * Decimal(self.para.T) / Decimal(44)
        # 물의 증기압(p_h2o)을 포함한 총 가스 압력(p_gas)을 계산
        p_gas = p_h2 + p_ch4 + p_co2 + Decimal(self.para.p_h2o)
        # 가스 압력과 물리적 매개변수를 기반으로 총 가스 유량(q_gas)을 계산
        q_gas = Decimal(self.para.k_p) * (p_gas - self.p_atm) * p_gas / self.p_atm
        # 출력 배열(y)을 0으로 초기화
        y = np.zeros(44)
        # 첫 34개의 상태 변수를 출력 배열에 그대로 복사
        y[:34] = x[:34]
        # 계산된 결과값들을 출력 배열의 나머지 공간에 추가
        y[34] = float(S_nh4_i)  # 암모늄 농도
        y[35] = float(S_co2)  # 이산화탄소 농도
        y[36] = float(phi)  # phi 값
        y[37] = float(S_H)  # 수소 이온 농도
        y[38] = float(pH)  # pH 값
        y[39] = float(p_h2)  # 수소 가스 부분 압력
        y[40] = float(p_ch4)  # 메탄 가스 부분 압력
        y[41] = float(p_co2)  # 이산화탄소 가스 부분 압력
        y[42] = float(p_gas)  # 총 가스 압력
        y[43] = float(q_gas)  # 총 가스 유량

        # 최종 계산 결과 배열을 반환
        return y

    def interp_input(self, t):
        """
        현재 시간 t에 대해 입력 데이터를 보간하여 반환하는 함수.

        Args:
            t (float): 현재 시간.

        Returns:
            tuple: 현재 시간에 해당하는 입력 데이터.
        """
        # 시간 배열 가져오기 (원본 데이터의 시간 값 유지)
        time = self.input_data.time

        # t보다 작거나 같은 가장 가까운 시간의 인덱스를 찾음
        closest_index = max(i for i, t_val in enumerate(time) if t_val <= t)

        # 해당 인덱스에서 입력 데이터를 추출
        time = self.input_data.time[closest_index]  # 현재 시점의 시간 값
        q_in = self.input_data.q_in[closest_index]  # 유입량
        S_su = self.input_data.S_su[closest_index]  # 가용성 당 농도
        S_aa = self.input_data.S_aa[closest_index]  # 아미노산 농도
        S_fa = self.input_data.S_fa[closest_index]  # 지방산 농도
        S_va = self.input_data.S_va[closest_index]  # 발레르산 농도
        S_bu = self.input_data.S_bu[closest_index]  # 부티르산 농도
        S_pro = self.input_data.S_pro[closest_index]  # 프로피온산 농도
        S_ac = self.input_data.S_ac[closest_index]  # 아세트산 농도
        S_h2 = self.input_data.S_h2[closest_index]  # 수소 농도
        S_ch4 = self.input_data.S_ch4[closest_index]  # 메탄 농도
        S_IC = self.input_data.S_IC[closest_index]  # 무기 탄소 농도
        S_IN = self.input_data.S_IN[closest_index]  # 무기 질소 농도
        S_h2o = self.input_data.S_h2o[closest_index]  # 물 농도
        X_ch = self.input_data.X_ch[closest_index]  # 탄수화물 농도
        X_pr = self.input_data.X_pr[closest_index]  # 단백질 농도
        X_li = self.input_data.X_li[closest_index]  # 지질 농도
        X_su = self.input_data.X_su[closest_index]  # 당 분해성 미생물 농도
        X_aa = self.input_data.X_aa[closest_index]  # 아미노산 분해성 미생물 농도
        X_fa = self.input_data.X_fa[closest_index]  # 지방산 분해성 미생물 농도
        X_va = self.input_data.X_va[closest_index]  # 발레르산 분해성 미생물 농도
        X_bu = self.input_data.X_bu[closest_index]  # 부티르산 분해성 미생물 농도
        X_pro = self.input_data.X_pro[closest_index]  # 프로피온산 분해성 미생물 농도
        X_ac = self.input_data.X_ac[closest_index]  # 아세트산 분해성 미생물 농도
        X_h2 = self.input_data.X_h2[closest_index]  # 수소 농도와 관련된 미생물
        S_cat = self.input_data.S_cat[closest_index]  # 양이온 농도
        S_an = self.input_data.S_an[closest_index]  # 음이온 농도
        S_va_i = self.input_data.S_va_i[closest_index]  # 발레르산 이온 농도
        S_bu_i = self.input_data.S_bu_i[closest_index]  # 부티르산 이온 농도
        S_pro_i = self.input_data.S_pro_i[closest_index]  # 프로피온산 이온 농도
        S_ac_i = self.input_data.S_ac_i[closest_index]  # 아세트산 이온 농도
        S_hco3_i = self.input_data.S_hco3_i[closest_index]  # 중탄산 이온 농도
        S_nh3 = self.input_data.S_nh3[closest_index]  # 암모니아 농도
        S_gas_h2 = self.input_data.S_gas_h2[closest_index]  # 수소 가스 농도
        S_gas_ch4 = self.input_data.S_gas_ch4[closest_index]  # 메탄 가스 농도
        S_gas_co2 = self.input_data.S_gas_co2[closest_index]  # 이산화탄소 가스 농도

        # 현재 시간의 모든 입력 데이터를 반환
        return (time, q_in, S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_h2o, X_ch, X_pr,
                X_li, X_su, X_aa, X_fa, X_va, X_bu, X_pro, X_ac, X_h2, S_cat, S_an, S_va_i, S_bu_i, S_pro_i, S_ac_i,
                S_hco3_i, S_nh3, S_gas_h2, S_gas_ch4, S_gas_co2)

    @staticmethod
    def get_x_from_initial(initial_val):
        """
        초기 상태 값을 기반으로 시스템 상태 변수 배열을 생성하는 함수.

        Args:
            initial_val (object): 초기 상태 값을 포함하는 객체. 객체의 속성은 시스템의 초기 조건을 나타냄.

        Returns:
            list: 초기 상태 변수의 리스트.
        """
        # 초기 상태 변수 리스트 생성
        x = [
            initial_val.S_su,  # 가용성 당 농도
            initial_val.S_aa,  # 아미노산 농도
            initial_val.S_fa,  # 지방산 농도
            initial_val.S_va,  # 발레르산 농도
            initial_val.S_bu,  # 부티르산 농도
            initial_val.S_pro,  # 프로피온산 농도
            initial_val.S_ac,  # 아세트산 농도
            initial_val.S_h2,  # 수소 농도
            initial_val.S_ch4,  # 메탄 농도
            initial_val.S_IC,  # 무기 탄소 농도
            initial_val.S_IN,  # 무기 질소 농도
            initial_val.S_h2o,  # 물 농도
            initial_val.X_ch,  # 탄수화물 농도
            initial_val.X_pr,  # 단백질 농도
            initial_val.X_li,  # 지질 농도
            initial_val.X_su,  # 당 분해성 미생물 농도
            initial_val.X_aa,  # 아미노산 분해성 미생물 농도
            initial_val.X_fa,  # 지방산 분해성 미생물 농도
            initial_val.X_va,  # 발레르산 분해성 미생물 농도
            initial_val.X_bu,  # 부티르산 분해성 미생물 농도
            initial_val.X_pro,  # 프로피온산 분해성 미생물 농도
            initial_val.X_ac,  # 아세트산 분해성 미생물 농도
            initial_val.X_h2,  # 수소 농도와 관련된 미생물
            initial_val.S_cat,  # 양이온 농도
            initial_val.S_an,  # 음이온 농도
            initial_val.S_va_i,  # 발레르산 이온 농도
            initial_val.S_bu_i,  # 부티르산 이온 농도
            initial_val.S_pro_i,  # 프로피온산 이온 농도
            initial_val.S_ac_i,  # 아세트산 이온 농도
            initial_val.S_hco3_i,  # 중탄산 이온 농도
            initial_val.S_nh3,  # 암모니아 농도
            initial_val.S_gas_h2,  # 수소 가스 농도
            initial_val.S_gas_ch4,  # 메탄 가스 농도
            initial_val.S_gas_co2  # 이산화탄소 가스 농도
        ]
        # 초기 상태 변수 리스트 반환
        return x

    def adm1_mass(self, t, x):
        """
        Anaerobic Digestion Model No. 1 (ADM1)의 질량 기반 구현 함수

        Args:
            t (float): 현재 시간.
            x (np.ndarray): 상태 변수 배열. 농도와 기타 매개변수를 포함.

        Returns:
            np.ndarray: ADM1 질량 기반 방정식을 사용해 계산된 상태 변수의 미분 값(dx/dt).
        """

        # **1단계: 입력 변수 보간**
        # 현재 시간 `t`에 해당하는 입력 변수들을 보간하여 필요한 값을 가져옴
        time, q_in, S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_h2o, X_ch, X_pr, \
        X_li, X_su, X_aa, X_fa, X_va, X_bu, X_pro, X_ac, X_h2, S_cat, S_an, S_va_i, S_bu_i, S_pro_i, S_ac_i, \
        S_hco3_i, S_nh3, S_gas_h2, S_gas_ch4, S_gas_co2 = self.interp_input(t)

        # **2단계: 상태 변수와 보조 값 계산**
        # 상태 변수 배열을 Decimal로 변환하여 고정 소수점 연산 수행
        x = [Decimal(val) for val in x]
        S_nh4_i = x[10] - x[30]  # 암모늄 이온의 농도 계산
        S_co2 = x[9] - x[29]  # 이산화탄소 농도 계산
        phi = (x[23] + S_nh4_i / Decimal(17) - x[29] / Decimal(44) - x[28] / Decimal(60)
               - x[27] / Decimal(74) - x[26] / Decimal(88) - x[25] / Decimal(102) - x[24])  # 전하 균형 계산

        # S_H는 양성자 농도로, pH 계산의 주요 변수
        S_H = -phi * Decimal(0.5) + Decimal(0.5) * (phi * phi + Decimal(4) * Decimal(self.para.K_w)).sqrt()

        # **3단계: 추가적인 상태 변수 계산**
        # pH 계산 (로그 함수 사용)
        pH = -S_H.ln() / Decimal(np.log(10))

        # 기체 압력 계산 (수소, 메탄, 이산화탄소)
        p_h2 = x[31] * Decimal(self.para.R) * Decimal(self.para.T) / Decimal(2)
        p_ch4 = x[32] * Decimal(self.para.R) * Decimal(self.para.T) / Decimal(16)
        p_co2 = x[33] * Decimal(self.para.R) * Decimal(self.para.T) / Decimal(44)
        p_gas = p_h2 + p_ch4 + p_co2 + Decimal(self.para.p_h2o)  # 총 기체 압력

        # 기체 생성량 계산
        q_gas = Decimal(self.para.k_p) * (p_gas - Decimal(self.p_atm)) * p_gas / Decimal(self.p_atm)

        # **억제 함수(inhibition functions) 정의**
        # 다양한 생물학적 반응의 억제 메커니즘을 정의. 이들은 반응 속도를 조정하거나 제한하는 데 중요한 역할을 함.
        inhibition = [
            x[10] / (x[10] + Decimal(self.para.K_I_IN)),  # 암모늄 이온에 의한 억제
            Decimal(self.para.K_I_fa) / (Decimal(self.para.K_I_fa) + x[7]),  # 지방산에 의한 억제
            Decimal(self.para.K_I_c4) / (Decimal(self.para.K_I_c4) + x[7]),  # 지방산 계열(c4)의 억제
            Decimal(self.para.K_I_pro) / (Decimal(self.para.K_I_pro) + x[7]),  # 프로피온산에 의한 억제
            # 산도(pH)에 의한 아미노산의 억제
            Decimal(10) ** (-(Decimal(3) / (Decimal(self.para.pK_u_aa) - Decimal(self.para.pK_l_aa))) *
                            (Decimal(self.para.pK_l_aa) + Decimal(self.para.pK_u_aa)) / Decimal(2)) /
            (S_H ** (Decimal(3) / (Decimal(self.para.pK_u_aa) - Decimal(self.para.pK_l_aa))) +
             Decimal(10) ** (-(Decimal(3) / (Decimal(self.para.pK_u_aa) - Decimal(self.para.pK_l_aa))) *
                             (Decimal(self.para.pK_l_aa) + Decimal(self.para.pK_u_aa)) / Decimal(2))),
            # 산도(pH)에 의한 아세트산의 억제
            Decimal(10) ** (-(Decimal(3) / (Decimal(self.para.pK_u_ac) - Decimal(self.para.pK_l_ac))) *
                            (Decimal(self.para.pK_l_ac) + Decimal(self.para.pK_u_ac)) / Decimal(2)) /
            (S_H ** (Decimal(3) / (Decimal(self.para.pK_u_ac) - Decimal(self.para.pK_l_ac))) +
             Decimal(10) ** (-(Decimal(3) / (Decimal(self.para.pK_u_ac) - Decimal(self.para.pK_l_ac))) *
                             (Decimal(self.para.pK_l_ac) + Decimal(self.para.pK_u_ac)) / Decimal(2))),
            # 산도(pH)에 의한 수소의 억제
            Decimal(10) ** (-(Decimal(3) / (Decimal(self.para.pK_u_h2) - Decimal(self.para.pK_l_h2))) *
                            (Decimal(self.para.pK_l_h2) + Decimal(self.para.pK_u_h2)) / Decimal(2)) /
            (S_H ** (Decimal(3) / (Decimal(self.para.pK_u_h2) - Decimal(self.para.pK_l_h2))) +
             Decimal(10) ** (-(Decimal(3) / (Decimal(self.para.pK_u_h2) - Decimal(self.para.pK_l_h2))) *
                             (Decimal(self.para.pK_l_h2) + Decimal(self.para.pK_u_h2)) / Decimal(2))),
            Decimal(self.para.K_I_nh3) / (Decimal(self.para.K_I_nh3) + x[30])  # 암모니아에 의한 억제
        ]

        # **반응 속도 정의** (Decimal 활용)
        # ADM1 모델에서 사용하는 다양한 반응 속도를 계산하여 rate 리스트에 저장합니다.
        # Decimal을 사용하여 소수점 계산의 정확도를 유지합니다.
        rate = [Decimal(0)] * 29  # 반응 속도를 저장할 리스트를 29개의 0으로 초기화

        # 셀룰로스를 분해하는 미생물의 반응 속도를 계산합니다.
        rate[0] = Decimal(self.para.k_ch) * Decimal(x[12])
        # 단백질을 분해하는 미생물의 반응 속도를 계산합니다.
        rate[1] = Decimal(self.para.k_pr) * Decimal(x[13])
        # 지질을 분해하는 미생물의 반응 속도를 계산합니다.
        rate[2] = Decimal(self.para.k_li) * Decimal(x[14])
        # 설탕과 같은 단순 탄수화물을 발효시키는 속도를 계산합니다.
        rate[3] = (Decimal(self.para.k_m_su) * Decimal(x[0]) / (Decimal(self.para.K_su) + Decimal(x[0])) *
                   Decimal(x[15]) * Decimal(inhibition[0]) * Decimal(inhibition[4]))
        # 아미노산을 발효시키는 미생물의 반응 속도를 계산합니다.
        rate[4] = (Decimal(self.para.k_m_aa) * Decimal(x[1]) / (Decimal(self.para.K_aa) + Decimal(x[1])) *
                   Decimal(x[16]) * Decimal(inhibition[0]) * Decimal(inhibition[4]))
        # 지방산을 발효시키는 반응 속도를 계산합니다.
        rate[5] = (Decimal(self.para.k_m_fa) * Decimal(x[2]) / (Decimal(self.para.K_fa) + Decimal(x[2])) *
                   Decimal(x[17]) * Decimal(inhibition[0]) * Decimal(inhibition[1]) * Decimal(inhibition[4]))
        # 프로피온산을 생성하는 반응 속도를 계산합니다.
        rate[6] = (Decimal(self.para.k_m_va) * Decimal(x[3]) / (Decimal(self.para.K_va) + Decimal(x[3])) *
                   Decimal(x[18]) * Decimal(x[3]) / (Decimal(x[4]) + Decimal(x[3]) + Decimal('1e-8')) *
                   Decimal(inhibition[0]) * Decimal(inhibition[2]) * Decimal(inhibition[4]))
        # 부티르산을 생성하는 반응 속도를 계산합니다.
        rate[7] = (Decimal(self.para.k_m_bu) * Decimal(x[4]) / (Decimal(self.para.K_bu) + Decimal(x[4])) *
                   Decimal(x[19]) * Decimal(x[4]) / (Decimal(x[3]) + Decimal(x[4]) + Decimal('1e-8')) *
                   Decimal(inhibition[0]) * Decimal(inhibition[2]) * Decimal(inhibition[4]))
        # 프로피온산이 소비되는 반응 속도를 계산합니다.
        rate[8] = (Decimal(self.para.k_m_pro) * Decimal(x[5]) / (Decimal(self.para.K_pro) + Decimal(x[5])) *
                   Decimal(x[20]) * Decimal(inhibition[0]) * Decimal(inhibition[3]) * Decimal(inhibition[4]))
        # 아세트산이 소비되는 반응 속도를 계산합니다.
        rate[9] = (Decimal(self.para.k_m_ac) * Decimal(x[6]) / (Decimal(self.para.K_ac) + Decimal(x[6])) *
                   Decimal(x[21]) * Decimal(inhibition[0]) * Decimal(inhibition[5]) * Decimal(inhibition[7]))
        # 수소를 소비하는 반응 속도를 계산합니다.
        rate[10] = (Decimal(self.para.k_m_h2) * Decimal(x[7]) / (Decimal(self.para.K_h2) + Decimal(x[7])) *
                    Decimal(x[22]) * Decimal(inhibition[0]) * Decimal(inhibition[6]))
        # **미생물 분해 반응 속도**
        # 미생물의 분해 속도를 계산하여 해당 리스트에 추가합니다.
        rate[11] = Decimal(self.para.k_dec) * Decimal(x[15])  # 당 발효 미생물 분해 속도
        rate[12] = Decimal(self.para.k_dec) * Decimal(x[16])  # 아미노산 미생물 분해 속도
        rate[13] = Decimal(self.para.k_dec) * Decimal(x[17])  # 지방산 미생물 분해 속도
        rate[14] = Decimal(self.para.k_dec) * Decimal(x[18])  # 프로피온산 미생물 분해 속도
        rate[15] = Decimal(self.para.k_dec) * Decimal(x[19])  # 부티르산 미생물 분해 속도
        rate[16] = Decimal(self.para.k_dec) * Decimal(x[20])  # 프로피온산 미생물 분해 속도
        rate[17] = Decimal(self.para.k_dec) * Decimal(x[21])  # 아세트산 미생물 분해 속도
        rate[18] = Decimal(self.para.k_dec) * Decimal(x[22])  # 수소 소비 미생물 분해 속도
        # **산-염기 반응 속도**
        # 다양한 산과 염기의 농도를 기준으로 반응 속도를 계산합니다.
        rate[19] = (Decimal(self.para.k_AB_va) *
                    (Decimal(x[25]) * (Decimal(self.para.K_a_va) + Decimal(S_H)) -
                     Decimal(self.para.K_a_va) * Decimal(x[3])))  # 프로피온산의 산-염기 반응 속도
        rate[20] = (Decimal(self.para.k_AB_bu) *
                    (Decimal(x[26]) * (Decimal(self.para.K_a_bu) + Decimal(S_H)) -
                     Decimal(self.para.K_a_bu) * Decimal(x[4])))  # 부티르산의 산-염기 반응 속도
        rate[21] = (Decimal(self.para.k_AB_pro) *
                    (Decimal(x[27]) * (Decimal(self.para.K_a_pro) + Decimal(S_H)) -
                     Decimal(self.para.K_a_pro) * Decimal(x[5])))  # 프로피온산의 산-염기 반응 속도
        rate[22] = (Decimal(self.para.k_AB_ac) *
                    (Decimal(x[28]) * (Decimal(self.para.K_a_ac) + Decimal(S_H)) -
                     Decimal(self.para.K_a_ac) * Decimal(x[6])))  # 아세트산의 산-염기 반응 속도
        rate[23] = (Decimal(self.para.k_AB_co2) *
                    (Decimal(x[29]) * (Decimal(self.para.K_a_co2) + Decimal(S_H)) -
                     Decimal(self.para.K_a_co2) * Decimal(x[9])))  # 이산화탄소의 산-염기 반응 속도
        rate[24] = (Decimal(self.para.k_AB_IN) *
                    (Decimal(x[30]) * (Decimal(self.para.K_a_IN) + Decimal(S_H)) -
                     Decimal(self.para.K_a_IN) * Decimal(x[10])))  # 암모니아의 산-염기 반응 속도
        # **기체 교환 반응 속도**
        # 수소, 메탄, 이산화탄소의 기체 교환 속도를 계산합니다.
        rate[25] = Decimal(self.para.k_La) * (Decimal(x[7]) - Decimal(2) *
                                              (Decimal(self.para.K_H_h2) * Decimal(p_h2)))  # 수소의 기체 교환 속도
        rate[26] = Decimal(self.para.k_La) * (Decimal(x[8]) - Decimal(16) *
                                              (Decimal(self.para.K_H_ch4) * Decimal(p_ch4)))  # 메탄의 기체 교환 속도
        rate[27] = Decimal(self.para.k_La) * (Decimal(S_co2) - Decimal(44) *
                                              (Decimal(self.para.K_H_co2) * Decimal(p_co2)))  # 이산화탄소의 기체 교환 속도

        # Define process equations with Decimal
        process = [Decimal(0)] * 34
        # **반응 프로세스 정의**
        # 각 반응 프로세스를 정의하여 시스템 내에서의 물질 및 기체 이동을 설명합니다.
        # 1번 프로세스: 발효 반응을 통한 당 분해 및 물질 흐름
        process[0] = (Decimal('1.1111') * rate[0] + Decimal('0.13482') * rate[2] -
                      Decimal('13.2724') * rate[3])
        # 2번 프로세스: 아미노산 분해 및 관련 물질 반응
        process[1] = rate[1] - Decimal('11.5665') * rate[4]
        # 3번 프로세스: 지방산 분해 및 물질의 이동
        process[2] = Decimal('0.95115') * rate[2] - Decimal('8.2136') * rate[5]
        # 4번 프로세스: 프로피온산 생성 및 반응
        process[3] = Decimal('1.8371') * rate[4] - Decimal('11.5757') * rate[6]
        # 5번 프로세스: 부티르산과 관련된 반응과 이동
        process[4] = (Decimal('0.91131') * rate[3] + Decimal('2.3289') * rate[4] -
                      Decimal('12.9817') * rate[7])
        # 6번 프로세스: 지방산 분해를 통한 주요 물질 생성
        process[5] = (Decimal('2.2734') * rate[3] + Decimal('0.53795') * rate[4] +
                      Decimal('7.9149') * rate[6] - Decimal('23.3892') * rate[8])
        # 7번 프로세스: 아세트산, 메탄 등의 주요 물질 흐름
        process[6] = (Decimal('4.8975') * rate[3] + Decimal('6.1053') * rate[4] +
                      Decimal('14.5554') * rate[5] + Decimal('6.4459') * rate[6] +
                      Decimal('16.6347') * rate[7] + Decimal('18.1566') * rate[8] -
                      Decimal('26.5447') * rate[9])
        # 8번 프로세스: 수소 기체 및 관련 물질의 반응과 이동
        process[7] = (Decimal('0.30475') * rate[3] + Decimal('0.12297') * rate[4] +
                      Decimal('0.83761') * rate[5] + Decimal('0.41881') * rate[6] +
                      Decimal('0.55841') * rate[7] + Decimal('1.8392') * rate[8] -
                      Decimal('2.9703') * rate[10] - rate[25])
        # 9번 프로세스: 메탄 기체 및 물질의 반응 흐름
        process[8] = Decimal('6.7367') * rate[9] + Decimal('5.5548') * rate[10] - rate[26]
        # 10번 프로세스: 이산화탄소 및 주요 기체 이동
        process[9] = (-Decimal('0.02933') * rate[2] + Decimal('4.4571') * rate[3] +
                      Decimal('2.8335') * rate[4] - Decimal('0.72457') * rate[5] -
                      Decimal('0.55945') * rate[6] - Decimal('0.38907') * rate[7] +
                      Decimal('13.1283') * rate[8] + Decimal('18.4808') * rate[9] -
                      Decimal('17.1839') * rate[10] - rate[27])
        # 11번 프로세스: pH 및 기타 미량 성분 변화
        process[10] = (-Decimal('0.15056') * rate[3] + Decimal('2.1033') * rate[4] -
                       Decimal('0.15056') * sum(rate[5:11]))
        # 12번 프로세스: 기체 생성 및 물질 축적 변화
        process[11] = (-Decimal('0.1111') * rate[0] - Decimal('0.05664') * rate[2] -
                       Decimal('0.4211') * rate[3] - Decimal('5.3025') * rate[4] -
                       Decimal('7.3043') * rate[5] - Decimal('3.4939') * rate[6] -
                       Decimal('4.6718') * rate[7] - Decimal('10.5843') * rate[8] +
                       Decimal('0.47776') * rate[9] + Decimal('13.75') * rate[10])
        # 13번 프로세스: 미생물 분해 반응 누적 변화
        process[12] = -rate[0] + Decimal('0.18') * sum(rate[11:19])
        # 14번 프로세스: 아미노산 관련 반응 누적 변화
        process[13] = -rate[1] + Decimal('0.77') * sum(rate[11:19])
        # 15번 프로세스: 지방산 분해 반응 누적 변화
        process[14] = -rate[2] + Decimal('0.05') * sum(rate[11:19])
        # **추가된 반응 프로세스 정의**
        # 특정 반응이나 물질의 생성 및 소멸 과정을 나타냄
        # 16번 프로세스: 당 분해 반응에 따른 미생물 작용
        process[15] = rate[3] - rate[11]
        # 17번 프로세스: 아미노산 분해 과정에 따른 변화
        process[16] = rate[4] - rate[12]
        # 18번 프로세스: 지방산 분해와 관련된 반응
        process[17] = rate[5] - rate[13]
        # 19번 프로세스: 프로피온산 분해에 따른 반응 흐름
        process[18] = rate[6] - rate[14]
        # 20번 프로세스: 부티르산 분해를 통한 물질 이동
        process[19] = rate[7] - rate[15]
        # 21번 프로세스: 프로피온산의 변환 과정
        process[20] = rate[8] - rate[16]
        # 22번 프로세스: 아세트산과 관련된 물질의 반응
        process[21] = rate[9] - rate[17]
        # 23번 프로세스: 수소 기체 반응에 따른 변화
        process[22] = rate[10] - rate[18]
        # 24번~25번 프로세스: 비활성 상태를 유지하는 단계
        process[23] = Decimal(0)  # 비활성 반응 (데이터 누락 또는 무작위 초기화)
        process[24] = Decimal(0)  # 비활성 반응
        # 26번~30번 프로세스: 특정 물질의 반응 손실에 따른 감소
        process[25] = -rate[19]  # 부피 반응 손실
        process[26] = -rate[20]  # 부티르산 반응 손실
        process[27] = -rate[21]  # 프로피온산 반응 손실
        process[28] = -rate[22]  # 아세트산 반응 손실
        process[29] = -rate[23]  # CO2 반응 손실
        # 31번~33번 프로세스: 액상-기체 상호작용 반응
        process[30] = -rate[24]  # NH3 반응 손실
        process[31] = (Decimal(self.V_liq) / Decimal(self.V_gas)) * rate[25]  # 수소의 액체-기체 이동
        process[32] = (Decimal(self.V_liq) / Decimal(self.V_gas)) * rate[26]  # 메탄의 액체-기체 이동
        process[33] = (Decimal(self.V_liq) / Decimal(self.V_gas)) * rate[27]  # CO2의 액체-기체 이동

        # **미분 방정식 정의**
        # ADM1 모델 내에서 각 상태 변수의 변화율 (dx/dt)을 계산

        dx = [Decimal(0)] * len(x)  # Decimal 타입으로 상태 변수 초기화
        # 1번 상태 변수 (S_su): 가용성 기질 농도의 변화율
        # 입력 유량(q_in)을 통해 유입된 기질과 현재 농도의 차이를 계산하고, 반응 과정을 포함하여 변화율을 계산
        dx[0] = Decimal(q_in) * (Decimal(S_su) - Decimal(x[0])) / Decimal(self.V_liq) + process[0]
        # 2번 상태 변수 (S_aa): 아미노산 농도의 변화율
        dx[1] = Decimal(q_in) * (Decimal(S_aa) - Decimal(x[1])) / Decimal(self.V_liq) + process[1]
        # 3번 상태 변수 (S_fa): 지방산 농도의 변화율
        dx[2] = Decimal(q_in) * (Decimal(S_fa) - Decimal(x[2])) / Decimal(self.V_liq) + process[2]
        # 4번 상태 변수 (S_va): 휘발성 지방산(Valerate) 농도의 변화율
        dx[3] = Decimal(q_in) * (Decimal(S_va) - Decimal(x[3])) / Decimal(self.V_liq) + process[3]
        # 5번 상태 변수 (S_bu): 부티레이트(Butyrate) 농도의 변화율
        dx[4] = Decimal(q_in) * (Decimal(S_bu) - Decimal(x[4])) / Decimal(self.V_liq) + process[4]
        # 6번 상태 변수 (S_pro): 프로피온산(Propionate) 농도의 변화율
        dx[5] = Decimal(q_in) * (Decimal(S_pro) - Decimal(x[5])) / Decimal(self.V_liq) + process[5]
        # 7번 상태 변수 (S_ac): 아세트산(Acetate) 농도의 변화율
        dx[6] = Decimal(q_in) * (Decimal(S_ac) - Decimal(x[6])) / Decimal(self.V_liq) + process[6]
        # 8번 상태 변수 (S_h2): 수소(H2) 농도의 변화율
        dx[7] = Decimal(q_in) * (Decimal(S_h2) - Decimal(x[7])) / Decimal(self.V_liq) + process[7]
        # 9번 상태 변수 (S_ch4): 메탄(CH4) 농도의 변화율
        dx[8] = Decimal(q_in) * (Decimal(S_ch4) - Decimal(x[8])) / Decimal(self.V_liq) + process[8]
        # 10번 상태 변수 (S_IC): 무기 탄소 (Inorganic Carbon) 농도의 변화율
        dx[9] = Decimal(q_in) * (Decimal(S_IC) - Decimal(x[9])) / Decimal(self.V_liq) + process[9]
        # 11번 상태 변수 (S_IN): 무기 질소 (Inorganic Nitrogen) 농도의 변화율
        dx[10] = Decimal(q_in) * (Decimal(S_IN) - Decimal(x[10])) / Decimal(self.V_liq) + process[10]
        # 12번 상태 변수 (S_h2o): 물(H2O) 농도의 변화율
        dx[11] = Decimal(q_in) * (Decimal(S_h2o) - Decimal(x[11])) / Decimal(self.V_liq) + process[11]
        # 13번 상태 변수 (X_ch): 탄수화물 분해 미생물 농도의 변화율
        dx[12] = Decimal(q_in) * (Decimal(X_ch) - Decimal(x[12])) / Decimal(self.V_liq) + process[12]
        # 14번 상태 변수 (X_pr): 단백질 분해 미생물 농도의 변화율
        dx[13] = Decimal(q_in) * (Decimal(X_pr) - Decimal(x[13])) / Decimal(self.V_liq) + process[13]
        # 15번 상태 변수 (X_li): 지방 분해 미생물 농도의 변화율
        dx[14] = Decimal(q_in) * (Decimal(X_li) - Decimal(x[14])) / Decimal(self.V_liq) + process[14]
        # 16번 상태 변수 (X_su): 당(Sugar) 분해 미생물 농도의 변화율
        dx[15] = Decimal(q_in) * (Decimal(X_su) - Decimal(x[15])) / Decimal(self.V_liq) + process[15]
        # 17번 상태 변수 (X_aa): 아미노산 분해 미생물 농도의 변화율
        dx[16] = Decimal(q_in) * (Decimal(X_aa) - Decimal(x[16])) / Decimal(self.V_liq) + process[16]
        # 18번 상태 변수 (X_fa): 지방산 분해 미생물 농도의 변화율
        dx[17] = Decimal(q_in) * (Decimal(X_fa) - Decimal(x[17])) / Decimal(self.V_liq) + process[17]
        # 19번 상태 변수 (X_va): 휘발성 지방산 분해 미생물 농도의 변화율
        dx[18] = Decimal(q_in) * (Decimal(X_va) - Decimal(x[18])) / Decimal(self.V_liq) + process[18]
        # 20번 상태 변수 (X_bu): 부티레이트 분해 미생물 농도의 변화율
        dx[19] = Decimal(q_in) * (Decimal(X_bu) - Decimal(x[19])) / Decimal(self.V_liq) + process[19]
        # 21번 상태 변수 (X_pro): 프로피온산 분해 미생물 농도의 변화율
        dx[20] = Decimal(q_in) * (Decimal(X_pro) - Decimal(x[20])) / Decimal(self.V_liq) + process[20]
        # 22번 상태 변수 (X_ac): 아세트산 분해 미생물 농도의 변화율
        dx[21] = Decimal(q_in) * (Decimal(X_ac) - Decimal(x[21])) / Decimal(self.V_liq) + process[21]
        # 23번 상태 변수 (X_h2): 수소 분해 미생물 농도의 변화율
        dx[22] = Decimal(q_in) * (Decimal(X_h2) - Decimal(x[22])) / Decimal(self.V_liq) + process[22]
        # 24번 상태 변수 (S_cat): 양이온 농도의 변화율
        dx[23] = Decimal(q_in) * (Decimal(S_cat) - Decimal(x[23])) / Decimal(self.V_liq) + process[23]
        # 25번 상태 변수 (S_an): 음이온 농도의 변화율
        dx[24] = Decimal(q_in) * (Decimal(S_an) - Decimal(x[24])) / Decimal(self.V_liq) + process[24]
        # 26번 상태 변수 (기체 분압 관련 값 계산 시작)
        # 기체 분압 변화율 계산 (고정된 프로세스)
        dx[25] = process[25]  # 휘발성 지방산에 해당하는 항목
        dx[26] = process[26]  # 부티레이트에 해당하는 항목
        dx[27] = process[27]  # 프로피온산에 해당하는 항목
        dx[28] = process[28]  # 아세트산에 해당하는 항목
        dx[29] = process[29]  # CO2 관련 항목
        dx[30] = process[30]  # 무기 질소 농도와 관련된 항목
        # 기체 상에서의 농도 변화율 (기체 부피와 유량에 의존)
        dx[31] = -Decimal(x[31]) * Decimal(q_gas) / Decimal(self.V_gas) + process[31]  # 수소 농도
        dx[32] = -Decimal(x[32]) * Decimal(q_gas) / Decimal(self.V_gas) + process[32]  # 메탄 농도
        dx[33] = -Decimal(x[33]) * Decimal(q_gas) / Decimal(self.V_gas) + process[33]  # CO2 농도

        # **호환성을 위해 dx를 float 타입으로 변환**
        # 결과를 numpy 배열로 반환하여 후속 계산에 사용
        return np.array([float(val) for val in dx])

