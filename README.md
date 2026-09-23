# News Recommender System

## Problem

iPrint is a media company that shows news, sports, weather, health and other articles to its users through an app. Earlier it only recommended popular or similar articles, and users were not finding this relevant. Because of this the company was slowly losing users.

The idea here is to build a recommendation system that can:
1. Suggest 10 new articles to a user when they open the app.
2. Suggest 10 similar articles once a user clicks on any article.

Articles that are removed from the app or already seen by the user should not be recommended again. Only English articles are used for content-based recommendations.

## Data Used

Two files were used for this project:

- **consumer_transanctions.csv** — has user interactions like watched, liked, commented, saved, followed.
- **platform_content.csv** — has article details like title, description, language, country.

Since there was no direct rating column, ratings were created manually based on interaction type:

| Interaction | Rating given |
|---|---|
| content_followed | 5 |
| content_commented_on | 4 |
| content_saved | 3 |
| content_liked | 2 |
| content_watched | 1 |

## What was done

- Removed articles that were pulled out from the app.
- Kept only English language articles for content based filtering.
- Removed duplicate rows.
- Did basic EDA — checked which country has most users, most common interaction type, correlation between numeric columns.

## Approach

Four different recommendation approaches were tried:

**1. Content Based Filtering**
Used the article description column, converted it into keywords, built a TF-IDF model using gensim, and used cosine similarity to find articles that are similar to each other in content.

**2. Collaborative Filtering (User Based)**
Built a user–item matrix using ratings, calculated similarity between users, and recommended articles that similar users liked.

**3. Collaborative Filtering (Item Based)**
Same idea but similarity was calculated between items (articles) instead of users.

**4. ALS Model**
Used the `implicit` library to build a matrix factorization model (Alternating Least Squares) on a sparse user-item matrix.

**5. Hybrid Models**
Combined scores from two models at a time (after normalizing them) to get a final ranked list:
- Item + Content
- Item + ALS
- ALS + Content
- User + ALS

## Evaluation

Used MAE and RMSE to check how good the collaborative filtering predictions were compared to actual test data.

- MAE ≈ 7.10
- RMSE ≈ 15.57

For a real product, click through rate (CTR) can be used to check performance in production — if a user keeps ignoring a certain type of article, similar articles should stop being recommended.

## Functions in the notebook

| Function | What it does |
|---|---|
| `cont_fil(item_id)` | content based recommendations for an article |
| `user_fil(user_id)` | collaborative filtering recommendation for a user |
| `item_fil(item_id)` | collaborative filtering recommendation for an article |
| `als_user(user_id)` | ALS based recommendation for a user |
| `als_item(item_id)` | ALS based recommendation for an article |
| `hybrid_itemco(item_id)` | hybrid of item + content |
| `hybrid_itemALS(item_id)` | hybrid of item + ALS |
| `hybrid_contALS(item_id)` | hybrid of content + ALS |
| `hybrid_userALS(user_id)` | hybrid of user + ALS |

## Tools Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- NLTK
- Gensim (TF-IDF, similarity matrix)
- Scikit-learn (train test split, pairwise distance, evaluation metrics)
- Implicit (ALS model)

## How to run

1. Keep `consumer_transanctions.csv` and `platform_content.csv` in the same folder as the notebook.
2. Install the required libraries (pandas, numpy, matplotlib, seaborn, nltk, gensim, scikit-learn, implicit, scipy).
3. Run the notebook cell by cell from top to bottom.

## Notes

- This was built as a learning/case study project, not a production system.
- Data used is from a public dataset for news recommendation, not real iPrint data.
