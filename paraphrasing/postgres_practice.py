import postgresql

command = ""
"""
CREATE TABLE related_words (
    query_word      text,
    synonyms        text[],
    antonyms        text[]
);

INSERT INTO related_words
    VALUES (word,
    '{}',
    '{}');
  
UPDATE related_words
SET synonyms = array_append(synonyms, synonym)
WHERE query_word = word;

UPDATE related_words
SET synonyms = array_append(antonyms, antonym)
WHERE query_word = word;
    
SELECT synonyms, antonyms 
FROM related_words
WHERE query_word = word;

-----------------------------
CREATE TABLE related_words (
    word            text,
    synonyms        text[],
    antonyms        text[]
);

INSERT INTO related_words
    VALUES ('like',
    '{}',
    '{}');

INSERT INTO related_words
    VALUES ('dog',
    '{}',
    '{}');
  
UPDATE related_words
SET synonyms = array_append(synonyms, 'love')
WHERE word = 'like';
    
SELECT * FROM related_words;
"""

db = postgresql.open('db.sqlite3')
db.execute("""CREATE TABLE related_words (
    query_word      text,
    synonyms        text[],
    antonyms        text[]
);""")
with db.xact():
    print("Worked")