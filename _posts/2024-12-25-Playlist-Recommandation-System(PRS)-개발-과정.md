20241113 윤기완
# [Demo](https://playlistrecommandationsystem.streamlit.app/)
---
# 개발 동기
 &nbsp;&nbsp;&nbsp;&nbsp;나는 **유튜브 플레이리스트**를 듣는 것을 좋아한다. 여기서 '유튜브 플레이리스트'란 단순히 음악을 모아놓은 재생목록이 아닌 하나의 종합예술로써의 진지함을 갖고 있는 콘텐츠를 의미한다. 이러한 콘텐츠는 아래의 요소를 포함한다.

- 개인의 사상이나 감정을 표현
- 음악과 시각적인 문화 요소를 배치
- 모든 구성 요소를 하나의 문구로 표현한 작품

ex) https://youtu.be/hs31WzyBIIg?si=e2i52ioz80i_Tl2W

 &nbsp;&nbsp;&nbsp;&nbsp;그럼에도 제일 큰 비중이 차지하는 것은 여전히 음악이다. 내가 개인적으로 선호하는 플레이리스트는 가사가 없는 음악, 뉴에이지, 영화음악, ASMR/엠비언스 같은 음악을 포함하는 플레이리스트이다. 이러한 플레이리스트를 선호하는 이유는 내가 주체가 되어서 음악을 즐길 수 있기 때문이다. '내가 주체가 된다'는 것은 음악에서 가수나 악기 같이 내가 아닌 존재가 주인공으로써 노래하고 연주되는 것이 아닌 청취자의 현재 상황과 감정 상태, 생각들이 노래와 어울려 하나의 음악, 혹은 그것을 감상하고 있는 순간을 완성한다는 의미이다. 이러한 점에서 나는 플레이리스트를 배경으로 나의 일과 독서, 휴식, 산책 등의 생활을 색칠하는 것을 즐긴다. 즉, 나의 일상을 영화 속 한 장면으로 만들어 몰입하고 더 풍부하게 만들어주는 음악 콘텐츠를 좋아한다.

&nbsp;&nbsp;&nbsp;&nbsp;하지만 음악만 해도 장르와 분위기가 다양한데 이러한 음악들을 엮어놓은 플레이리스트를 한 두 개 키워드로 표현하는 것은 여간 쉽지 않다. 따라서 플레이리스트, 특히 유튜브 플레이리스트 생태계에서는 채널의 운영자, 플레이리스트를 제작하는 채널이하나의 장르와 분위기을 내포하는 키워드로써 작용한다. 이러한 점을 활용해 나는 개인이 선호하는 플레이리스트와 유사한 플레이리스트를 찾을 수 있도록, 선호하는 채널과 유사한 다른 채널을 추천해주는 서비스를 기획하게 되었다.

&nbsp;&nbsp;&nbsp;&nbsp;본 서비스은 2024년도 2학기에 수강한 *Data & AI*와 *기초빅데이터프로그래밍* 수업을 수강하면서 학습한 지식을 바탕으로 제작하였다. 데이터 수집과 전처리는 *기초빅데이터프로그래밍*에서 학습한 File I/O과 정규식을 통해 구현하였고, 이를 바탕으로 채널을 추천해주는 모델은 *Data & AI*에서 학습한 Collaborative Filtering을 기반으로 개인적으로 추가 학습을 진행해 Content-based-Filtering을 통해 구현하였다. 모든 과정은 `Python`을 사용하여 구현하였다.

&nbsp;&nbsp;&nbsp;&nbsp;이러한 구상이 가능했던 것은 사전에 497개의 플레이리스트 유튜브 채널 데이터를 분석한 *빅데이터 기반 뮤직 플레이리스트 매칭 플랫폼* [플플](https://plpl.kr/)의 데이터 랩 덕분이었다. 해당 자료는 플플 사이트 내에서 자유롭게 사용할 수 있고, 상업적 용도로 배포, 재판매될 수 없다고 이용약관에 명시되어 있다.![image](/assets/resource/2024-12-25-Playlist-Recommandation-SystemPRS-개발-과정/스크린샷-2024-12-18-17.51.55.png)본 프로젝트는 해당 자료를 크롤링 프레임워크인 `selenium`을 통해 추출하여 추천 시스템의 입력 데이터로 사용하였으나, 상업적 목적을 위해 사용하지 않았음을 밝힌다.

그럼 PRS를 구현하는 과정의 핵심 부분들을 차례로 소개하겠다.



# 데이터 수집
![image](/assets/resource/2024-12-25-Playlist-Recommandation-SystemPRS-개발-과정/Pasted-image-20241218180255.png)
&nbsp;&nbsp;&nbsp;&nbsp;위 사진은 데이터를 수집하는 과정을 간략하게 나타낸다. `selenium`을 통해 크롤링을 진행하였는데 핵심 원리는 `HTML`의 `XPATH`를 이중 `for`문 그리고 `if`문으로 추출해내는 것이다.
```python
list_of_channels = []

for i in tqdm(range(497)):
	channel = {}
	if i == 0: # 첫 번째 행
		for j in range(7):
			if j == 0: # 채널명
				element = driver.find_element(By.XPATH, f'/html/body/div[2]/div[3]/div[1]/div/div/div[{i+1}]/div[2]/div[1]/div[2]/div')
			else:		
				element = driver.find_element(By.XPATH, f'/html/body/div[2]/div[3]/div[1]/div/div/div[{i+1}]/div[2]/div[{j+1}]/div')
			data = element.text
			if isinstance(data, str) and data.replace(',', '').isdigit():
				data = int(data.replace(',', ''))
			channel[j] = data
```

나는 크롬 개발자 도구를 통해 추출해야하는 데이터의 `XPATH`를 하나씩 추출해 패턴을 발견할 수 있었다. 

> `XPATH`: [확장 생성 언어](https://ko.wikipedia.org/wiki/XML "XML") 문서의 구조를 통해 경로 위에 지정한 구문을 사용하여 항목을 배치하고 처리하는 방법을 기술하는 언어

당시 나에게는 `XPATH`가 클라이언트에게 보여지는 화면에서 해당 데이터의 위치를 상대적으로 나타내는 데이터로서 다가와 해당 웹페이지의 전체 구조를 모르는 입장에서 유용한 값으로 사용할 수 있었다. 

```
0 channel_names = '/html/body/div[2]/div[3]/div[1]/div/div/div[1]/div[2]/div[1]/div[2]/div'
1 subscribers = '/html/body/div[2]/div[3]/div[1]/div/div/div[1]/div[2]/div[2]/div'
2 video_num = '/html/body/div[2]/div[3]/div[1]/div/div/div[1]/div[2]/div[3]/div'
3 total_view = '/html/body/div[2]/div[3]/div[1]/div/div/div[1]/div[2]/div[4]/div'
4 vs_index = '/html/body/div[2]/div[3]/div[1]/div/div/div[1]/div[2]/div[5]/div'
5 vv_index = '/html/body/div[2]/div[3]/div[1]/div/div/div[1]/div[2]/div[6]/div'
6 cpi = '/html/body/div[2]/div[3]/div[1]/div/div/div[1]/div[2]/div[7]/div'

####################################

0 channel_names = '/html/body/div[2]/div[3]/div[1]/div/div/div[2]/div/div[1]/div[2]/div'
1 subscribers = '/html/body/div[2]/div[3]/div[1]/div/div/div[2]/div/div[2]/div'
2 video_num = '/html/body/div[2]/div[3]/div[1]/div/div/div[2]/div/div[3]/div'

0 channel_names = '/html/body/div[2]/div[3]/div[1]/div/div/div[3]/div/div[1]/div[2]/div'
1 subscribers = '/html/body/div[2]/div[3]/div[1]/div/div/div[3]/div/div[2]/div'
```

그렇게 `XPATH`의 패턴을 분석한 결과 전체적인 table 구조 안에서 인덱스 값을 1씩 증가시키면 한 행(채널)씩 순회하며 각 값들을 뽑아낼 수 있었다. 이를 이중 `for`문을 통해 구현하였고 성공적으로 데이터를 추출할 수 있었다.

그런데 여기서 발생한 문제가 바로 '채널의 상세 페이지 내에 있는 `genre`와 `mood` 데이터를 어떻게 뽑을 것이냐'였다. 이전에 추출한 데이터는 한 페이지 내에 모두 로드되어 이중 `for`문을 통해 추출할 수 있었지만 상세 데이터는 해당 채널을 클릭해서 동적으로 로드되는 창 내에 위치하기에 한 번에 추출할 수 없을 듯이 보였다.
![[스크린샷 2024-12-23 20.40.31.png|채널 상세 페이지 예시]]
하지만 다음 코드를 통해 문제를 해결할 수 있었다.
```python
to_detail_page = driver.find_element(By.XPATH, f'/html/body/div[2]/div[3]/div[1]/div/div/div[{i+1}]/div[2]/div[1]/div[2]/div')

to_detail_page.click()
```
이는 웹페이지의 특정 부분을 클릭할 수 있도록 하는 코드이다. 이와 함께 `time.sleep()` 코드도 사용하여 특정 부분을 클릭해 상세 페이지를 로드시키고 위에서 설명한 것과 같은 방법으로 데이터를 추출할 수 있었다.

그렇게 추출한 한 채널에 대한 데이터를 집합에 저장하고 이 집합들을 하나의 리스트로 저장하였다.
```
[{
0: '때껄룩ᴛᴀᴋᴇ ᴀ ʟᴏᴏᴋ', 
1: 1930000, 
2: 408, 
3: 708387927, 
4: 367, 
5: 1736245, 
6: 41120, 
'genres': ['Folk (국외)', 'Pop (국외)', 'R&B / Soul (국외)', 'Rap / Hip-hop (국외)', 'Blues (국내)', 'Blues (국내)', 'R&B / Soul (국내)', 'Rap / Hip-hop (국내)', 'Dance (국내)'], 
'moods': ['모든 분위기']
}]
```

# 데이터 전처리
## `csv`로 저장하기 위한
&nbsp;&nbsp;&nbsp;&nbsp;위에서 수집한 데이터를 저장하기 위해 `pandas`를 활용해 `csv`로 변환하여 저장하였다.
![image](/assets/resource/2024-12-25-Playlist-Recommandation-SystemPRS-개발-과정/스크린샷-2024-12-18-18.34.55.png)
이때, `genres`와 `moods` 데이터 중 불필요한 공백이 데이터 내에 포함되어 있어 다음 코드를 통해 공백을 제거해주었다.
```
# 공백 제거 함수

def clean_empty_strings(column):

return column.apply(
lambda x: [item.strip() for item in eval(x) if item.strip()] if isinstance(x, str) else x
)

# genres와 moods 열에서 공백 제거
df['genres'] = clean_empty_strings(df['genres'])
df['moods'] = clean_empty_strings(df['moods'])
```

그리고 `python`의 문법(리스트는 \[\], 문자열은 '')이 문자열 자체로 인식되어 csv에 저장되어서 이를 처리하고자 `', '.join(eval(x))` 코드를 통해 순수한 데이터로 변환시켜 저장하였다. `pandas`를 통해 여러 데이터에 특정 함수를 적용시켜 쉽게 전처리를 진행할 수 있었고 이를 통해 `pandas`의 강력함을 체험할 수 있었다.


## `csv`에서 다시 불러온 후
&nbsp;&nbsp;&nbsp;&nbsp;이후 모델을 학습시키기 위해 `csv`를 다시 python 자료형으로 불러오는 과정에서의 전처리는 다음과 같이 진행하였다.
- `isnull()`, `fillna('')`: 해당 부분에 데이터가 없는 경우, 공백으로 처리
- `if pd.isna(x) or x == '-' or x == '':`, `return float(x)`: `cpi` 데이터(`int`) 중 공백 문자열(`''`) 혹은 `-`로 누락된 데이터를 공백으로 처리, `int` 자료형일 경우, 추후 vectorization을 위해 `float`로 반환
- `make_list(x)`, `df[feature] = df[feature].apply(make_list)`: 문자열로 저장된 `genre`와 `mood` 데이터를 `list` 데이터로 묶어주는 사용자 함수를 만들고 적용
- `clean_data(x)`, `df[feature] = df[feature].apply(clean_data)`: `genres`와 `moods` 데이터에 추후 `BoW`를 적용하기 위해 `clean_data`(공백 제거, 대문자 $\rightarrow$ 소문자 ) 사용자 정의 함수를 만들고 적용

# 추천 모델 개발

### Feature Soup 만들기
&nbsp;&nbsp;&nbsp;&nbsp;이제 추천 모델을 개발하기 위해 데이터를 하나의 Soup로 만들어 모델에 학습시켜야했다. 주요로 사용한 데이터는 장르와 분위기, 그리고 CPI(Channel Power Index)이었다. 여기서 CPI를 제외한 데이터는 모두 문자열로 이루어져있었기에 BoW(Bag of Word)라는 자연어 처리 모형을 사용하였다.

> **단어 가방(bag-of-words)** 모형은 [자연어 처리](https://ko.wikipedia.org/wiki/%EC%9E%90%EC%97%B0%EC%96%B4_%EC%B2%98%EB%A6%AC "자연어 처리") 및 [정보 검색](https://ko.wikipedia.org/wiki/%EC%A0%95%EB%B3%B4_%EA%B2%80%EC%83%89 "정보 검색") (IR)에 사용되는 단순화 된 표현이다. 이 모델에서 텍스트 (문장 또는 문서 등)는 문법 및 단어 순서는 무시하고 다중성을 유지하면서 단어의 [가방(멀티 세트)](https://ko.wikipedia.org/wiki/%EC%A4%91%EB%B3%B5%EC%A7%91%ED%95%A9 "중복집합")으로 표시된다. 단어 가방 모델은 컴퓨터 비전에도 사용되었다.
> 출처: [위키백과](https://ko.wikipedia.org/wiki/%EB%8B%A8%EC%96%B4_%EA%B0%80%EB%B0%A9_%EB%AA%A8%ED%98%95)

아래 코드를 통해 각 채널에 해당하는 장르와 분위기 키워드를 `' '`(띄어쓰기)를 기준으로 하나의 문자열로 묶어 soup 데이터로 만들어 새로운 열에 추가하였다.
![image](/assets/resource/2024-12-25-Playlist-Recommandation-SystemPRS-개발-과정/스크린샷-2024-12-23-17.41.31.png)


## Soup Vectorization과 CPI Normalization
&nbsp;&nbsp;&nbsp;&nbsp;두 문자열 데이터를 하나의 soup로 묶은 이후 각 feature를 기준으로 벡터화를 진행하였다. 이를 통해 문자열 데이터였던 soup(장르와 분위기)는 `0`과 `1` 중 하나의 값인 벡터로 표현될 수 있다. 이는 외부 라이브러리인 `sklearn`의 `CountVectorizer`를 통해 구현하였다.
![image](/assets/resource/2024-12-25-Playlist-Recommandation-SystemPRS-개발-과정/스크린샷-2024-12-23-17.46.11.png)
아래 출력값을 통해 총  497개의 채널에 대해 48개의 feature 키워드에 대응하는 벡터화가 진행되었음을 알 수 있다.

&nbsp;&nbsp;&nbsp;&nbsp;그러나 문자열 데이터였던 장르와 분위기 feature와 다르게 정수 데이터인 CPI(Channel Index Power)은 BoW 방식을 사용할 수 없었다. 이를 해결하기 위해 위에서 벡터화된 결과 벡터에 마지막 한 열을 병합하여 CPI 데이터를 추가하는 방법을 선택하였다. 전체적인 구조는 다음과 같다.
![image](/assets/resource/2024-12-25-Playlist-Recommandation-SystemPRS-개발-과정/스크린샷-2024-12-23-17.50.40.png)
```Python
# CPI 정규화
from sklearn.preprocessing import MinMaxScaler

# NaN이나 공백을 0으로 대체
df['cpi_cleaned'] = df['cpi'].replace('', 0).astype(float)

# 정규화
scaler = MinMaxScaler()
df['cpi_normalized'] = scaler.fit_transform(df'cpi_cleaned')
```
여기서 CPI 데이터를 그대로 병합하게 될 경우, 같은 차원의 데이터라고 모델이 판단하지 못하기에 정규화를 통해 데이터의 차원을 통일시켰다. 위에서와 마찬가지로 `sklearn` 라이브러리의 `MinMaxScaler` 함수를 통해 정규화를 진행하였다.

```python
# 장르 & 분위기 matrix + cpi_normalized marix 결합

from scipy.sparse import hstack
import numpy as np

cpi_matrix = np.array(df['cpi_normalized']).reshape(-1, 1) # CPI 값을 2D 배열로 변환
final_matrix = hstack([count_matrix, cpi_matrix]) # soup와 CPI를 결합
```
이후 `scipy` 라이브러리의 `hstack` 함수를 통해 CPI 수평 병합을 진행해 각 채널의 벡터 데이터를 완성시켰다.



## 코사인 유사도 계산
위 과정을 통해 각 채널은 `[0, 1]`의 값으로 벡터화되었다. 이를 기반으로 서로 다른 채널의 유사도를 측정해 비슷한 채널을 추천하는 모델을 만들 수 있었다. 코사인 유사도를 사용하였고 `sklearn` 라이브러리의 `cosine_similarity`를 사용해 각 채널 벡터들끼리 계산하여 `cosine_similarity` 행렬을 구하였다.
![image](/assets/resource/2024-12-25-Playlist-Recommandation-SystemPRS-개발-과정/스크린샷-2024-12-23-17.59.20.png)
코사인 유사도 행렬의 대각선 성분들은 같은 채널에 대한 코사인 유사도이므로 `1`로 고정된다. 이 최종 코사인 유사도 행렬을 통해 각 채널에 대한 유사도를 정렬하면추천할 채널을 얻을 수 있다.


## Playlist Channel 추천
```python
# 유튜브 플레이리스트 채널의 이름을 입력받으면 코사인 유사도를 통해서 가장 유사도가 높은 상위 10개의 채널 목록 반환

def get_recommendations(channel_name, cosine_sim=cosine_sim):
	# 채널을 통해서 전체 데이터 기준 그 채널의 index 값을 얻기
	idx = indices[channel_name]
	
	# 코사인 유사도 매트릭스 (cosine_sim) 에서 idx 에 해당하는 데이터를 (idx, 유사도) 형태로 얻기
	sim_scores = list(enumerate(cosine_sim[idx]))
	
	# 코사인 유사도 기준으로 내림차순 정렬
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	
	# 자기 자신을 제외한 10개의 채널을 슬라이싱
	sim_scores = sim_scores[1:11]
	
	sim_scores_top10 = [ str(x[1]) for x in sim_scores ]
	print('\n'.join(sim_scores_top10))
	
	# 추천 채널 목록 10개의 인덱스 정보 추출
	channel_indices = [i[0] for i in sim_scores]
	
	# 인덱스 정보를 통해 채널 이름 추출
	return df['channel_name'].iloc[channel_indices]
```
채널 이름을 인덱스로 반환해주는 함수 `indices`와 위에서 구한 유사도 `cosine_sim`을 통해, 입력받은 채널과 유사한 10개의 채널을 추천해주는 함수를 만들었다. 내림차순으로 10개의 채널 이름과 인덱스, 그리고 코사인 유사도를 출력하도록 하였다.
![[스크린샷 2024-12-23 18.18.28.png|채널 추천 결과]]



# Web Publishing
&nbsp;&nbsp;&nbsp;&nbsp;위에서 완성한 플레이리스트 채널 추천 시스템을 모두가 사용할 수 있도록 하기 위해 `streamlit` framework를 통해 웹으로 공유하였다.

## pickle화
우선 `Python`의 `pickle` 모듈을 이용하여 채널 이름 데이터인`playlist_channels`와 코사인 유사도 `cosine_sim`을 직렬화하였다. 그리고 web publishing을 위한 python 파일을 새로 생성하여 역직렬화해 데이터를 불러왔다.
![image](/assets/resource/2024-12-25-Playlist-Recommandation-SystemPRS-개발-과정/스크린샷-2024-12-23-18.24.45.png)
```python
# app.py 파일에서

playlist_channels = pickle.load(open("playlist_channels.pickle", "rb"))
cosine_sim = pickle.load(open("cosine_sim.pickle", "rb"))
```


## `get_recommendations`함수 재구현
&nbsp;&nbsp;&nbsp;&nbsp;역직렬화를 한 데이터를 가지고 이전에 구현한 추천 함수를 웹사이트에 표시할 정보에 맞게 수정하여 다시 코드를 구현했다.
```python
def get_recommendations(channel_name):
	# 채널을 통해서 전체 데이터 기준 그 채널의 index 값을 얻기
	idx = playlist_channels[playlist_channels == channel_name].index[0]
	
	# 코사인 유사도 매트릭스 (cosine_sim) 에서 idx 에 해당하는 데이터를 (idx, 유사도) 형태로 얻기
	sim_scores = list(enumerate(cosine_sim[idx]))
	
	# 코사인 유사도 기준으로 내림차순 정렬
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	
	# 자기 자신을 제외한 10개의 채널을 슬라이싱
	sim_scores = sim_scores[1:11]
	
	# 추천 채널 목록 10개의 인덱스 정보 추출
	channel_indices = [i[0] for i in sim_scores]
	
	# 인덱스 정보를 통해 채널 이름 추출
	channel_names = []
	for i in channel_indices:
		channel_names.append(playlist_channels.iloc[i])
	
	return channel_names
```
이를 통해 추천할 채널의 이름을 리스트로 반환하는 함수로 재구현하였다.


## streamlit을 통한 publishing
```python
st.set_page_config(layout="wide")
st.header("Playlist Recommendation System")

# movie_list = movies["title"].values
playlist_channel_list = list(playlist_channels)
channel = st.selectbox("Choose a playlist channel you like", playlist_channel_list)

if st.button("Recommend"):
	with st.spinner("Please wait..."):
		channel_names = get_recommendations(channel)
		
		idx = 0
		for i in range(10):
			cols = st.columns(1)
			for col in cols:
				col.write(channel_names[idx])
				idx += 1
```
최종적으로 위 간략한 코드를 통해 웹의 형태로 출력할 수 있었다. 

## 최종 결과
&nbsp;&nbsp;&nbsp;&nbsp;이후 `streamlit Cloud` 서비스를 통해 쉽게 퍼블리싱을 할 수 있었다. 해당 서비스는 이 [링크](https://playlistrecommandationsystem.streamlit.app/)를 통해 접속할 수 있다.
![image](/assets/resource/2024-12-25-Playlist-Recommandation-SystemPRS-개발-과정/스크린샷-2024-12-23-18.32.06.png)



# 후기 및 추후 계획
&nbsp;&nbsp;&nbsp;&nbsp;이번 프로젝트는 학교 수업에서 배운 내용과 나의 관심사인 플레이리스트를 접목시키고자 한 시도에서 출발하였다. 내가 좋아하는 것을 프로젝트로 설정하면서 나또한 즐기면서 작업할 수 있었고, 관련이 없을 것 같은 데이터 과학과 플레이리스트를 추천 시스템이라는 연결점을 통해 구현하면서 코딩을 통해서 할 수 있는 것은 상상하기 나름이라는 것을 몸소 느낄 수 있었던 중요한 경험이었다. 기술이 발전하면서 생활의 모든 것이 빅데이터로 전환되고 플레이리스트라는 비교적 새로운 콘텐츠에 대해 벌써 데이터를 수집하고 무언가 시도하는 회사가 있다는 점이 놀라웠다. 그리고 그 데이터를 통해 나 또한 무언가를 만들어 냈다는 경험이 앞으로의 프로젝트를 설정하고 진행하는데 좋은 시작점이 되어준 것 같다.

&nbsp;&nbsp;&nbsp;&nbsp;`Python`에 대한 인식 또한 바뀌게 되었는데 그동안 큰 감흥없이 배운 기본적인 문법, 반복문, 조건문이 프레임워크와 라이브러리들과 결합해 어마어마한 확장성을 가질 수 있다는 것을 깨달을 수 있었다. 그리고 이렇게 활용할 수 있다는 것을 알게 해 준 머신러닝과 딥러닝의 이론적 접근과 정보 처리 등 여러가지 알고리즘적인 지식 또한 중요하다는 것을 몸소 느낄 수 있었다.
&nbsp;&nbsp;&nbsp;&nbsp;한 가지 아쉬운 점은 제대로된 머신 러닝을 적용해보지 못한 것이다. 이 프로젝트에서 모델은 스스로 학습하고 추천하는 것이 아닌 행렬 연산을 바탕으로 데이터를 정렬하고 그에 맞게 추천했다는 한계가 있어 인공지능이라고 하기에는 무리가 있다. 추후에는 기계가 스스로 학습하여 더 입체적인 추천을 해줄 수 있는 데이터와 모델, 서비스를 구상해 보고싶다.

&nbsp;&nbsp;&nbsp;&nbsp;앞으로는 현재로서 나의 목표인 '음악의 일상화'를 위해 코딩 능력과 인공지능/알고리즘 공부를 이어나갈 것이며 이와 더불어 나 또한 음악을 본격적으로 제작해보자 한다. 그 과정에서 내가 시도한 것들을 기록하고 공유할 것이며 내가 할 수 있는 것을 계속해서 모색해 나갈 것이다.