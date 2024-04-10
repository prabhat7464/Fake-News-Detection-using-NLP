df = pd.read_csv("News_dataset.csv")
df = df.dropna()
df.reset_index(inplace=True)
df = df.drop(['id','text','author'],axis = 1)

# Preprocess sample data
sample_data = 'The quick brown fox jumps over the lazy dog'
sample_data = sample_data.split()
sample_data = [data.lower() for data in sample_data]
stopwords = stopwords.words('english')
sample_data = [data for data in sample_data if data not in stopwords]
ps = PorterStemmer()
sample_data_stemming = [ps.stem(data) for data in sample_data]
lm = WordNetLemmatizer()
sample_data_lemma = [lm.lemmatize(data) for data in sample_data]

# Preprocess news titles
lm = WordNetLemmatizer()
corpus = []
for i in range(len(df)):
    review = re.sub('^a-zA-Z0-9',' ', df['title'][i])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(x) for x in review if x not in stopwords]
    review = " ".join(review)
    corpus.append(review)

# Vectorize text data

x = tf.fit_transform(corpus).toarray()
y = df['label']

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10, stratify=y)