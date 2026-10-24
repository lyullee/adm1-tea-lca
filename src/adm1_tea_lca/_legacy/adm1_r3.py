import numpy as np
from scipy.integrate import solve_ivp
from decimal import Decimal, getcontext


getcontext().prec = 50  # 소수점 50자리 정밀도


class ADM1R3:
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
            return self.adm1_r3(t, x)

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
        y_output = np.zeros((len(solution.t), 26))  # 상태 변수 출력 배열 초기화 (44개의 변수)

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
        S_nh4_i = Decimal(x[3]) - Decimal(x[14])
        # 액상에서 이산화탄소 농도(S_co2)를 계산
        S_co2 = Decimal(x[2]) - Decimal(x[13])
        # pH 균형에 관련된 대수 변수 phi를 계산
        phi = Decimal(x[10]) + S_nh4_i / Decimal(17) \
              - Decimal(x[13]) / Decimal(44) - Decimal(x[12]) / Decimal(60) \
              - Decimal(x[11])
        # phi 값과 물의 이온화 상수(K_w)를 사용하여 수소 이온 농도(S_H)를 계산
        S_H = -Decimal(phi) * Decimal(0.5) + Decimal(0.5) * Decimal(phi * phi + 4 * Decimal(self.para.K_w)).sqrt()
        # 수소 이온 농도를 사용하여 pH 값을 계산
        pH = -Decimal(S_H).log10()
        # 메탄 가스의 부분 압력(p_ch4)을 계산
        p_ch4 = Decimal(x[15]) * Decimal(self.para.R) * Decimal(self.para.T) / Decimal(16)
        # 이산화탄소 가스의 부분 압력(p_co2)을 계산
        p_co2 = Decimal(x[16]) * Decimal(self.para.R) * Decimal(self.para.T) / Decimal(44)
        # 물의 증기압(p_h2o)을 포함한 총 가스 압력(p_gas)을 계산
        p_gas = p_ch4 + p_co2 + Decimal(self.para.p_h2o)
        # 가스 압력과 물리적 매개변수를 기반으로 총 가스 유량(q_gas)을 계산
        q_gas = Decimal(self.para.k_p) * (p_gas - self.p_atm) * p_gas / self.p_atm

        # 출력 배열(y)을 0으로 초기화
        y = np.zeros(26)
        # 첫 34개의 상태 변수를 출력 배열에 그대로 복사
        y[:17] = x[:17]
        # 계산된 결과값들을 출력 배열의 나머지 공간에 추가
        y[17] = float(S_nh4_i)  # 암모늄 농도
        y[18] = float(S_co2)  # 이산화탄소 농도
        y[19] = float(phi)  # phi 값
        y[20] = float(S_H)  # 수소 이온 농도
        y[21] = float(pH)  # pH 값
        y[22] = float(p_ch4)  # 메탄 가스 부분 압력
        y[23] = float(p_co2)  # 이산화탄소 가스 부분 압력
        y[24] = float(p_gas)  # 총 가스 압력
        y[25] = float(q_gas)  # 총 가스 유량

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
        time = self.input_data.time[closest_index]
        q_in = self.input_data.q_in[closest_index]
        S_ac = self.input_data.S_ac[closest_index]
        S_ch4 = self.input_data.S_ch4[closest_index]
        S_IC = self.input_data.S_IC[closest_index]
        S_IN = self.input_data.S_IN[closest_index]
        S_h2o = self.input_data.S_h2o[closest_index]
        X_ch = self.input_data.X_ch[closest_index]
        X_pr = self.input_data.X_pr[closest_index]
        X_li = self.input_data.X_li[closest_index]
        X_bac = self.input_data.X_bac[closest_index]
        X_ac = self.input_data.X_ac[closest_index]
        S_cat = self.input_data.S_cat[closest_index]
        S_an = self.input_data.S_an[closest_index]
        S_ac_i = self.input_data.S_ac_i[closest_index]
        S_hco3_i = self.input_data.S_hco3_i[closest_index]
        S_nh3 = self.input_data.S_nh3[closest_index]
        S_gas_ch4 = self.input_data.S_gas_ch4[closest_index]
        S_gas_co2 = self.input_data.S_gas_co2[closest_index]

        # 현재 시간의 모든 입력 데이터를 반환
        return (time, q_in, S_ac, S_ch4, S_IC, S_IN, S_h2o, X_ch, X_pr, X_li, X_bac,
                X_ac, S_cat, S_an, S_ac_i, S_hco3_i, S_nh3, S_gas_ch4, S_gas_co2)

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
            initial_val.S_ac,
            initial_val.S_ch4,
            initial_val.S_IC,
            initial_val.S_IN,
            initial_val.S_h2o,
            initial_val.X_ch,
            initial_val.X_pr,
            initial_val.X_li,
            initial_val.X_bac,
            initial_val.X_ac,
            initial_val.S_cat,
            initial_val.S_an,
            initial_val.S_ac_i,
            initial_val.S_hco3_i,
            initial_val.S_nh3,
            initial_val.S_gas_ch4,
            initial_val.S_gas_co2
        ]
        # 초기 상태 변수 리스트 반환
        return x

    def adm1_r3(self, t, x):
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
        time, q_in, S_ac, S_ch4, S_IC, S_IN, S_h2o, X_ch, X_pr, X_li, X_bac,\
        X_ac, S_cat, S_an, S_ac_i, S_hco3_i, S_nh3, S_gas_ch4, S_gas_co2 = self.interp_input(t)

        # **2단계: 상태 변수와 보조 값 계산**
        # 상태 변수 배열을 Decimal로 변환하여 고정 소수점 연산 수행
        x = [Decimal(val) for val in x]
        S_nh4_i = x[3] - x[14]  # 암모늄 이온의 농도 계산
        S_co2 = x[2] - x[13]  # 이산화탄소 농도 계산
        phi = x[10] + S_nh4_i / Decimal(17) - x[13] / Decimal(44) - x[12] / Decimal(60) - x[11]  # 전하 균형 계산

        # S_H는 양성자 농도로, pH 계산의 주요 변수
        S_H = -phi * Decimal(0.5) + Decimal(0.5) * (phi * phi + Decimal(4) * Decimal(self.para.K_w)).sqrt()

        # **3단계: 추가적인 상태 변수 계산**
        # pH 계산 (로그 함수 사용)
        pH = -S_H.ln() / Decimal(np.log(10))

        # 기체 압력 계산 (수소, 메탄, 이산화탄소)
        p_ch4 = x[15] * Decimal(self.para.R) * Decimal(self.para.T) / Decimal(16)
        p_co2 = x[16] * Decimal(self.para.R) * Decimal(self.para.T) / Decimal(44)
        p_gas = p_ch4 + p_co2 + Decimal(self.para.p_h2o)  # 총 기체 압력

        # 기체 생성량 계산
        q_gas = Decimal(self.para.k_p) * (p_gas - Decimal(self.p_atm)) * p_gas / Decimal(self.p_atm)

        # **억제 함수(inhibition functions) 정의**
        # 다양한 생물학적 반응의 억제 메커니즘을 정의. 이들은 반응 속도를 조정하거나 제한하는 데 중요한 역할을 함.
        inhibition = [
            x[3] / (x[3] + Decimal(self.para.K_I_IN)),  # 암모늄 이온에 의한 억제
            # 산도(pH)에 의한 아세트산의 억제
            Decimal(10) ** (-(Decimal(3) / (Decimal(self.para.pK_u_ac) - Decimal(self.para.pK_l_ac))) *
                            (Decimal(self.para.pK_l_ac) + Decimal(self.para.pK_u_ac)) / Decimal(2)) /
            (S_H ** (Decimal(3) / (Decimal(self.para.pK_u_ac) - Decimal(self.para.pK_l_ac))) +
             Decimal(10) ** (-(Decimal(3) / (Decimal(self.para.pK_u_ac) - Decimal(self.para.pK_l_ac))) *
                             (Decimal(self.para.pK_l_ac) + Decimal(self.para.pK_u_ac)) / Decimal(2))),
            Decimal(self.para.K_I_nh3) / (Decimal(self.para.K_I_nh3) + x[14])  # 암모니아에 의한 억제
        ]

        # **반응 속도 정의** (Decimal 활용)
        # ADM1 모델에서 사용하는 다양한 반응 속도를 계산하여 rate 리스트에 저장합니다.
        # Decimal을 사용하여 소수점 계산의 정확도를 유지합니다.
        # 반응 속도 정의
        rate = [Decimal(0)] * 11  # 반응 속도를 저장할 리스트를 초기화
        # 반응 속도 계산
        rate[0] = Decimal(self.para.k_ch) * Decimal(x[5])  # MATLAB: rate(1) = k_ch * x(6);
        rate[1] = Decimal(self.para.k_pr) * Decimal(x[6])  # MATLAB: rate(2) = k_pr * x(7);
        rate[2] = Decimal(self.para.k_li) * Decimal(x[7])  # MATLAB: rate(3) = k_li * x(8);
        rate[3] = (Decimal(self.para.k_m_ac) * Decimal(x[0]) / (Decimal(self.para.K_ac) + Decimal(x[0])) *
                   Decimal(x[9]) * Decimal(inhibition[0]) * Decimal(inhibition[1]) * Decimal(inhibition[2]))
        rate[4] = Decimal(self.para.k_dec) * Decimal(x[8])  # MATLAB: rate(5) = k_dec * x(9);
        rate[5] = Decimal(self.para.k_dec) * Decimal(x[9])  # MATLAB: rate(6) = k_dec * x(10);
        rate[6] = (Decimal(self.para.k_AB_ac) * (Decimal(x[12]) * (Decimal(self.para.K_a_ac) + Decimal(S_H)) -
                                                 Decimal(self.para.K_a_ac) * Decimal(x[0])))
        rate[7] = (Decimal(self.para.k_AB_co2) * (Decimal(x[13]) * (Decimal(self.para.K_a_co2) + Decimal(S_H)) -
                                                  Decimal(self.para.K_a_co2) * Decimal(x[2])))
        rate[8] = (Decimal(self.para.k_AB_IN) * (Decimal(x[14]) * (Decimal(self.para.K_a_IN) + Decimal(S_H)) -
                                                 Decimal(self.para.K_a_IN) * Decimal(x[3])))
        rate[9] = (Decimal(self.para.k_La) * (
                    Decimal(x[1]) - Decimal(16) * (Decimal(self.para.K_H_ch4) * Decimal(p_ch4))))
        rate[10] = (Decimal(self.para.k_La) * (
                    Decimal(S_co2) - Decimal(44) * (Decimal(self.para.K_H_co2) * Decimal(p_co2))))

        # 프로세스 정의
        process = [Decimal(0)] * 18  # 프로세스 배열 초기화, 총 18개 요소
        # 프로세스 계산
        process[0] = (Decimal('0.6555') * rate[0] + Decimal('0.9947') * rate[1] +
                      Decimal('1.7651') * rate[2] - Decimal('26.5447') * rate[3])
        process[1] = (Decimal('0.081837') * rate[0] + Decimal('0.069636') * rate[1] +
                      Decimal('0.19133') * rate[2] + Decimal('6.7367') * rate[3] - rate[9])
        process[2] = (Decimal('0.2245') * rate[0] + Decimal('0.10291') * rate[1] -
                      Decimal('0.64716') * rate[2] + Decimal('18.4808') * rate[3] - rate[10])
        process[3] = (-Decimal('0.016932') * rate[0] + Decimal('0.17456') * rate[1] -
                      Decimal('0.024406') * rate[2] - Decimal('0.15056') * rate[3])
        process[4] = (-Decimal('0.057375') * rate[0] - Decimal('0.47666') * rate[1] -
                      Decimal('0.44695') * rate[2] + Decimal('0.4778') * rate[3])
        process[5] = (-rate[0] + Decimal('0.18') * rate[4] + Decimal('0.18') * rate[5])
        process[6] = (-rate[1] + Decimal('0.77') * rate[4] + Decimal('0.77') * rate[5])
        process[7] = (-rate[2] + Decimal('0.05') * rate[4] + Decimal('0.05') * rate[5])
        process[8] = (Decimal('0.11246') * rate[0] + Decimal('0.13486') * rate[1] +
                      Decimal('0.1621') * rate[2] - rate[4])
        process[9] = rate[3] - rate[5]
        process[10] = Decimal(0)
        process[11] = Decimal(0)
        process[12] = -rate[6]
        process[13] = -rate[7]
        process[14] = -rate[8]
        process[15] = (Decimal(self.V_liq) / Decimal(self.V_gas)) * rate[9]
        process[16] = (Decimal(self.V_liq) / Decimal(self.V_gas)) * rate[10]

        # **미분 방정식 정의**
        # ADM1 모델 내에서 각 상태 변수의 변화율 (dx/dt)을 계산

        dx = [Decimal(0)] * 17  # 상태 변수 초기화

        dx[0] = Decimal(q_in) * (Decimal(S_ac) - Decimal(x[0])) / Decimal(self.V_liq) + Decimal(process[0])
        dx[1] = Decimal(q_in) * (Decimal(S_ch4) - Decimal(x[1])) / Decimal(self.V_liq) + Decimal(process[1])
        dx[2] = Decimal(q_in) * (Decimal(S_IC) - Decimal(x[2])) / Decimal(self.V_liq) + Decimal(process[2])
        dx[3] = Decimal(q_in) * (Decimal(S_IN) - Decimal(x[3])) / Decimal(self.V_liq) + Decimal(process[3])
        dx[4] = Decimal(q_in) * (Decimal(S_h2o) - Decimal(x[4])) / Decimal(self.V_liq) + Decimal(process[4])
        dx[5] = Decimal(q_in) * (Decimal(X_ch) - Decimal(x[5])) / Decimal(self.V_liq) + Decimal(process[5])
        dx[6] = Decimal(q_in) * (Decimal(X_pr) - Decimal(x[6])) / Decimal(self.V_liq) + Decimal(process[6])
        dx[7] = Decimal(q_in) * (Decimal(X_li) - Decimal(x[7])) / Decimal(self.V_liq) + Decimal(process[7])
        dx[8] = Decimal(q_in) * (Decimal(X_bac) - Decimal(x[8])) / Decimal(self.V_liq) + Decimal(process[8])
        dx[9] = Decimal(q_in) * (Decimal(X_ac) - Decimal(x[9])) / Decimal(self.V_liq) + Decimal(process[9])
        dx[10] = Decimal(q_in) * (Decimal(S_cat) - Decimal(x[10])) / Decimal(self.V_liq) + Decimal(process[10])
        dx[11] = Decimal(q_in) * (Decimal(S_an) - Decimal(x[11])) / Decimal(self.V_liq) + Decimal(process[11])
        dx[12] = Decimal(process[12])
        dx[13] = Decimal(process[13])
        dx[14] = Decimal(process[14])
        dx[15] = -Decimal(x[15]) * Decimal(q_gas) / Decimal(self.V_gas) + Decimal(process[15])
        dx[16] = -Decimal(x[16]) * Decimal(q_gas) / Decimal(self.V_gas) + Decimal(process[16])

        return np.array([float(val) for val in dx])

