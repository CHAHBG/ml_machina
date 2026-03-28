import json
import re

file_path = '/Users/user/Desktop/Data_Analyst_Courses/MLsupervised/EDA_Maintenance_Predictive.ipynb'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Regex list matches variations of common emoticons, symbols, emojis found in the script
    emoji_pattern = re.compile(
        r'['
        r'\U0001f600-\U0001f64f'  # emoticons
        r'\U0001f300-\U0001f5ff'  # symbols & pictographs
        r'\U0001f680-\U0001f6ff'  # transport & map symbols
        r'\U0001f1e0-\U0001f1ff'  # flags (iOS)
        r'\U00002702-\U000027b0'
        r'\U000024c2-\U0001f251'
        r'🔧✅📐💾📊⚠️📌🔗🛠️🤖🏆'
        r']+', flags=re.UNICODE)

    def clean_text(text):
        text = emoji_pattern.sub('', text)
        # Remove em-dash U+2014, and specifically the one in the text
        text = text.replace('—', '')
        # Remove extra space where em_dash was
        text = text.replace('  ', ' ')
        return text

    for cell in data.get('cells', []):
        if 'source' in cell:
            if isinstance(cell['source'], list):
                cell['source'] = [clean_text(line) for line in cell['source']]
            else:
                cell['source'] = clean_text(cell['source'])

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        
    print("Nettoyage réussi!")
except Exception as e:
    print(f"Erreur: {e}")
