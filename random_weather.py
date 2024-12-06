import random

def generate_weather_data(n=10):
    cities = ["台北", "新北", "桃園", "台中", "高雄", "台南", "基隆", "新竹", "嘉義", "宜蘭"]
    weather_conditions = ["晴天", "多雲", "陰天", "小雨", "大雨", "雷陣雨", "強風"]
    
    if n > len(cities):
        raise ValueError("要求生成的天氣資訊數量超過了可用的城市數量！")
    
    selected_cities = random.sample(cities, n)  # 隨機選擇不重複的城市
    weather_data = {}
    
    for city in selected_cities:
        condition = random.choice(weather_conditions)
        temperature = random.randint(20, 35)  # 隨機生成溫度範圍在 20°C ~ 35°C
        weather_data[city] = f"{condition}，溫度{temperature}°C"
    
    return weather_data
