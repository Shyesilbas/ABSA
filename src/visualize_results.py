import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(base_dir, 'data', 'final_report.csv')

if not os.path.exists(csv_path):
    print("CSV file not found")
    exit()

df = pd.read_csv(csv_path)

df.rename(columns={'conf': 'confidence_score', 'aspect': 'detected_aspect'}, inplace=True)
df.columns = df.columns.str.strip()

print(f"Columns found: {df.columns.tolist()}")

if 'confidence_score' in df.columns:
    df = df[df['confidence_score'] > 60]
else:
    print("Warning: Confidence score column not found.")

ignore_words = ['nin', 'un', 'in', 'yi', 'yı', 'su', 'bu', 'o', 'bir', 'sahibi', 'bakanı', 'GENEL', 'tane']
df_clean = df[~df['detected_aspect'].isin(ignore_words)]

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
colors = {"Negative": "#FF4B4B", "Neutral": "#7D7D7D", "Positive": "#4CAF50"}

try:
    print("\n" + "=" * 50)
    print("SENTIMENT DISTRIBUTION SUMMARY")
    print("=" * 50)
    sentiment_counts = df_clean['sentiment'].value_counts()
    print(sentiment_counts)
    print("-" * 30)
    total_tweets = len(df_clean)
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_tweets) * 100
        print(f"{sentiment.ljust(15)}: {count} ({percentage:.1f}%)")

    ax = sns.countplot(x="sentiment", data=df_clean, palette=colors, order=["Negative", "Neutral", "Positive"])
    plt.title(f'Sentiment distribution of analyzed {len(df_clean)} tweets', fontsize=15, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Total', fontsize=12)

    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

    save_path1 = os.path.join(base_dir, 'data', 'chart_sentiment_distribution.png')
    plt.savefig(save_path1, dpi=300)
    print(f"\nGraph 1 saved to: {save_path1}")

except Exception as e:
    print(f"Error while drawing the graph 1: {e}")

try:
    top_10_aspects = df_clean['detected_aspect'].value_counts().nlargest(10).index
    df_top10 = df_clean[df_clean['detected_aspect'].isin(top_10_aspects)]

    print("\n" + "=" * 50)
    print("TOP 10 DISCUSSED TOPICS & SENTIMENT BREAKDOWN")
    print("=" * 50)

    aspect_report = pd.crosstab(df_top10['detected_aspect'], df_top10['sentiment'])
    aspect_report['Total'] = aspect_report.sum(axis=1)
    aspect_report = aspect_report.sort_values('Total', ascending=False)

    print(aspect_report)
    print("-" * 50)

    plt.figure(figsize=(12, 8))
    sns.countplot(y="detected_aspect", hue="sentiment", data=df_top10,
                  order=top_10_aspects, palette=colors)
    plt.title('Top 10 Topics and Sentiment Distribution', fontsize=15, fontweight='bold')
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Aspect', fontsize=12)
    plt.legend(title='Sentiment State')

    save_path2 = os.path.join(base_dir, 'data', 'chart_top_aspects.png')
    plt.savefig(save_path2, dpi=300)
    print(f"Graph 2 saved to: {save_path2}")

except Exception as e:
    print(f"Error drawing the graph 2: {e}")

plt.show()
