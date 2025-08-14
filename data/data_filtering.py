import pandas as pd

# CSV 읽기
df = pd.read_csv("필터링_컬럼정리.csv", encoding="utf-8-sig", on_bad_lines="skip")

# '폐업일자'를 날짜형으로 변환 (NaT 처리 포함)
df['폐업일자'] = pd.to_datetime(df['폐업일자'], errors='coerce')

# 2000년 1월 1일 이전인 행 삭제
df = df[(df['폐업일자'].isna()) | (df['폐업일자'] >= '2000-01-01')]

# 결과 저장
df.to_csv("필터링_폐업일자정리.csv", index=False, encoding="utf-8-sig")

print("정리 완료! 2000년 이전 폐업 데이터가 제거되었습니다.")
