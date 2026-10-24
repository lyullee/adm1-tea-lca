import parameters  # 파라미터 값을 정의하는 모듈 임포트
import input  # 입력 데이터 값을 정의하는 모듈 임포트
import initial  # 초기 값 설정 모듈 임포트
from adm1_basic import ADM1  # ADM1 모델 클래스 임포트
from adm1_r1 import ADM1R1  # ADM1 모델 클래스 임포트
from adm1_r2 import ADM1R2  # ADM1 모델 클래스 임포트
from adm1_r3 import ADM1R3  # ADM1 모델 클래스 임포트
from adm1_r4 import ADM1R4  # ADM1 모델 클래스 임포트
import pandas as pd  # 데이터 프레임 및 엑셀 파일 처리를 위한 pandas 임포트


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    print("start up")  # 프로그램 시작 알림
    model = 'ADM1R4'
    v_liq = 100  # 액체 부피 설정
    v_gas = 10  # 기체 부피 설정
    time_range = (0, 100)  # 시뮬레이션 시간 범위 설정

    if model == 'ADM1':
        # ADM1 클래스의 파라미터 접근
        params = parameters.ADM1  # 파라미터 값 가져오기
        input_values = input.ADM1  # 입력 데이터 가져오기
        initial_values = initial.ADM1  # 초기 값 가져오기
        # ADM1 클래스 실행
        adm1_model = ADM1(v_liq, v_gas, time_range, input_values, initial_values, params)  # ADM1 모델 초기화
        # 결과 계산
        t_output, y_output = adm1_model.solve_adm1()  # 시뮬레이션 결과 계산
        output_filename = "ADM1_output_py.xlsx"  # 저장할 엑셀 파일 이름

    elif model == 'ADM1R1':
        params = parameters.ADM1R1  # 파라미터 값 가져오기
        input_values = input.ADM1R1  # 입력 데이터 가져오기
        initial_values = initial.ADM1R1  # 초기 값 가져오기
        # ADM1 클래스 실행
        adm1_model = ADM1R1(v_liq, v_gas, time_range, input_values, initial_values, params)  # ADM1 모델 초기화
        # 결과 계산
        t_output, y_output = adm1_model.solve_adm1()  # 시뮬레이션 결과 계산
        output_filename = "ADM1_output_py_r1.xlsx"  # 저장할 엑셀 파일 이름

    elif model == 'ADM1R2':
        params = parameters.ADM1R2  # 파라미터 값 가져오기
        input_values = input.ADM1R2  # 입력 데이터 가져오기
        initial_values = initial.ADM1R2  # 초기 값 가져오기
        # ADM1 클래스 실행
        adm1_model = ADM1R2(v_liq, v_gas, time_range, input_values, initial_values, params)  # ADM1 모델 초기화
        # 결과 계산
        t_output, y_output = adm1_model.solve_adm1()  # 시뮬레이션 결과 계산
        output_filename = "ADM1_output_py_r2.xlsx"  # 저장할 엑셀 파일 이름

    elif model == 'ADM1R3':
        params = parameters.ADM1R3  # 파라미터 값 가져오기
        input_values = input.ADM1R3  # 입력 데이터 가져오기
        initial_values = initial.ADM1R3  # 초기 값 가져오기
        # ADM1 클래스 실행
        adm1_model = ADM1R3(v_liq, v_gas, time_range, input_values, initial_values, params)  # ADM1 모델 초기화
        # 결과 계산
        t_output, y_output = adm1_model.solve_adm1()  # 시뮬레이션 결과 계산
        output_filename = "ADM1_output_py_r3.xlsx"  # 저장할 엑셀 파일 이름

    elif model == 'ADM1R4':
        params = parameters.ADM1R4  # 파라미터 값 가져오기
        input_values = input.ADM1R4  # 입력 데이터 가져오기
        initial_values = initial.ADM1R4  # 초기 값 가져오기
        # ADM1 클래스 실행
        adm1_model = ADM1R4(v_liq, v_gas, time_range, input_values, initial_values, params)  # ADM1 모델 초기화
        # 결과 계산
        t_output, y_output = adm1_model.solve_adm1()  # 시뮬레이션 결과 계산
        output_filename = "ADM1_output_py_r4.xlsx"  # 저장할 엑셀 파일 이름

    # 결과를 엑셀로 저장
    output_data = pd.DataFrame({  # 결과 데이터를 데이터프레임으로 변환
        "Time": t_output,  # 시간 데이터
        **{f"Output_{i}": y for i, y in enumerate(y_output.T, start=1)}  # 출력 데이터 열로 추가
    })
    output_data.to_excel(output_filename, index=False)  # 엑셀 파일로 저장

    print(f"Results saved to {output_filename}")  # 저장 완료 알림
