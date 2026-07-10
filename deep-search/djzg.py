from complete_real_analyzer import CompleteRealAnalyzer
import asyncio

async def analyze():
    analyzer = CompleteRealAnalyzer()
    result = await analyzer.complete_analysis("大金重工")
    return result

# 运行分析
asyncio.run(analyze())