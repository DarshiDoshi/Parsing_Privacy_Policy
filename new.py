import re
import os
import json
import numbers
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tqdm as tq

# Define base path and load classification regex
# base_path = r'D:\project-course\new'
with open(r'classification.json') as f:
    clsRegEx = json.load(f)
for cls, matches in clsRegEx.items():
    for i in range(len(matches)):
        if matches[i][-1] == '*':
            matches[i] = r'\b' + matches[i]
        else:
            matches[i] = r'\b' + matches[i] + r'\b'

# Function to calculate text similarity
def is_similar(header, target_keywords, threshold=0.7):
    corpus = [header] + target_keywords
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()
    similarity_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return any(score >= threshold for score in similarity_scores)

# Function to get preceding text of a table
def getPrevText(table):
    prev_ele = table
    final_text = ""
    table_text = table.text.strip()
    prev_texts = [table_text]
    prev_html = [str(table)]
    ctr = 0
    for _ in range(5):
        if ctr == 2:
            break
        temp_text = prev_ele.find_previous().text.strip()
        prev_html.append(str(prev_ele))
        prev_ele = prev_ele.find_previous()
        temp_len = 1000000
        for html in prev_html:
            temp_html = str(prev_ele).replace(html, '').strip()
            if len(temp_html) < temp_len:
                temp_len = len(temp_html)
                min_html = temp_html
                break
        if '<table' in min_html or '<tr' in min_html or '<td' in min_html or '<th' in min_html:
            break
        for prev_text in prev_texts:
            temp_text = temp_text.replace(prev_text, '').strip()
        if len(temp_text) > 0:
            final_text = temp_text + ". " + final_text
            prev_texts.append(temp_text)
            ctr += 1
    return final_text

# Function to classify cookies
def classifyCookieText(candidate):
    result = []
    for cls, matches in clsRegEx.items():
        for match in matches:
            res = re.search(match, candidate, re.IGNORECASE)
            if res:
                result.append([cls, match, candidate])
                break
    for i in range(len(result)):
        if result[i][0] == 'Necessary':
            result = [result[i]]
            break
    if len(result) == 0:
        result.append(['Unmatched', '', ''])
    return result

# Function to handle table processing
def runTableHandler(url, html):
    print(f"Processing URL: {url}")
    cls_predictions = []
    soup = BeautifulSoup(html, 'html.parser')
    table_htmls = soup.find_all('table')
    if len(table_htmls) == 0:
        print("No tables found in the page")
        return cls_predictions

    url_dict = {}
    url_list = []

    cookie_name_keywords = ["Name", "Name of Cookies", "cookie name", "cookies"]
    category_keywords = ["Category", "Type"]

    for table_html in table_htmls:
        table_df = pd.read_html(StringIO(str(table_html)))
        for table in table_df:
            # print(table)
            print("-------------------------------------------------")
            if isinstance(table.columns[0], numbers.Number):
                table.columns = table.iloc[0]
                table = table.reindex(table.index.drop(0))

            # Identify columns
            cookie_col = None
            for col in table.columns:
                if is_similar(str(col), cookie_name_keywords):
                    cookie_col = col
                    break

            category_col = None
            for col in table.columns:
                if is_similar(str(col), category_keywords):
                    category_col = col
                    break

            if cookie_col and category_col:
                print(f"Found 'Cookie Name' column: {cookie_col} and 'Category/Type' column: {category_col}")
                for it, row in table.iterrows():
                    # Classify using the category column
                    pred_cls = classifyCookieText(row[category_col])[0][0]
                    prev_text = ""
                    # If unmatched, use previous text to classify
                    if pred_cls == "Unmatched":
                        prev_text = getPrevText(table_html)
                        pred_cls = classifyCookieText(prev_text)[0][0]

                    # Store results
                    if pred_cls not in url_dict.keys():
                        url_dict[pred_cls] = {
                            'url': url,
                            'cookies': [row[cookie_col]],
                            'prevText': prev_text,
                            'prediction': pred_cls,
                            'match': row[category_col] if pred_cls != "Unmatched" else prev_text
                        }
                    else:
                        data = url_dict[pred_cls]
                        data['cookies'].append(row[cookie_col])
                        url_dict[pred_cls] = data

            elif cookie_col:
                print(f"Found 'Cookie Name' column: {cookie_col}, but 'Category/Type' column is missing")
                cookie_list = list(
                    table[cookie_col]
                    .str.split(',|/')
                    .explode()
                    .dropna()
                    .str.strip()
                    .unique()
                )
                prev_text = getPrevText(table_html)
                pred_cls = classifyCookieText(prev_text)
                for cls in pred_cls:
                    url_list.append({
                        'url': url,
                        'cookies': cookie_list,
                        'prevText': prev_text,
                        'prediction': cls[0],
                        'match': cls[1]
                    })
            else:
                print("No Cookie table found")

    if len(url_dict) > 0:
        cls_predictions += list(url_dict.values())
    if len(url_list) > 0:
        cls_predictions += url_list
    return cls_predictions

# Selenium setup
opts = webdriver.FirefoxOptions()
opts.headless = False
driver = webdriver.Firefox(options=opts)

urls = pd.read_csv('urls.csv')['url'].values

cls_predictions = []
for url in tq.tqdm(urls):
    driver.get(url)
    page = driver.page_source
    cls_predictions += runTableHandler(url, page)
driver.close()

# Save results to JSON and Excel
json_obj = json.dumps(cls_predictions, indent=4)
with open("cls_predictions.json", "w") as f:
    f.write(json_obj)
df = pd.DataFrame(cls_predictions)
df.to_excel('cls_predictions.xlsx', index=False)