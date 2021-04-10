from io import TextIOWrapper
from concurrent.futures import ProcessPoolExecutor

from nltk.util import bigrams, trigrams
from nltk import word_tokenize
import re

"""
Write a program that can correctly access in which language a string is written. Languages include:
1. Dutch 2. English 3. German 4. French 5. Spanish 6. Italian

Source word lists : https://wortschatz.uni-leipzig.de/en/download
"""

if __name__ == "__main__":
    # test strings
    dict_strings_to_test_program: dict[str, str] = {
        'dutch': "Ik ging op de fiets naar de winkel",
        'english': "I went to the grocery store by bike",
        'german': "Mein Deutsch ist sehr slecht und Sauerkraut mit Kartoffeln",
        'french': "Je ne sais pas monsieur je veux deux croissant",
        'spanish': "Luce el número cuatro en la espalda y el",
        'italian': "La manovra che abbiamo già annunciato, consistente e significativa"
    }

    # files that contain language sentences
    dict_paths_to_word_collections: dict[str, str] = {
        'german': "wordCollections/test/deu_news_2015_10K/deu_news_2015_10K-sentences.txt",
        'english': "wordCollections/test/eng_news_2015_10K/eng_news_2015_10K-sentences.txt",
        'french': "wordCollections/test/fra_news_2010_10K-text/fra_news_2010_10K-sentences.txt",
        'italian': "wordCollections/test/ita_news_2010_10K-text/ita_news_2010_10K-sentences.txt",
        'dutch': "wordCollections/test/nld_wikipedia_2016_10K/nld_wikipedia_2016_10K-sentences.txt",
        'spanish': "wordCollections/test/spa_news_2011_10K/spa_news_2011_10K-sentences.txt"
    }

    def sentence_to_list(sentence: str) -> list[str]:
        """
        Convert a sentence to a list that contains only the words that are in the sentence
        1. Remove digits from sentence
        2. Convert sentence to lowercase
        3. Remove commas, newline characters and tabs from the sentence
        4. Split sentence at every occurrence of a space
        5. Save the split sentence to a list
        :param sentence: sentence of which you want to extract the words and store those words in a list
        :return: list in which every element is a word from the sentence parameter
        """

        # return tuple whose elements are words from the sentence parameter
        return word_tokenize(re.sub(r"\d+", "", sentence).lower()
                    .replace(',', '')
                    .replace('\n', '')
                    .replace('\t', ''))


    def file_to_tuple(file_name: str) -> tuple[str]:
        """
        Take a string that corresponds to a text file. Iterate over the lines in the text file and perform
        'sentence_to_list' on the lines in the text file.
        :param file_name: string that specifies the path to the file
        :return: tuple[str]
        """
        # initially empty string that will eventually contain all of the strings that are found in
        # the file with the corresponding file_name argument
        sentence: str = ""

        # reading content of file and adding all of the lines of the file to 'sentence'
        file: TextIOWrapper = open(file_name, 'r')
        for line in file:
            sentence += line + " "

        # return tuple that contains all of the words in the file with the name of the file_name parameter
        return tuple(sentence_to_list(sentence))

    def tuple_to_ngrams(tuple_with_words: tuple[str], n=2) -> tuple[tuple[str]]:
        """
        Take a tuple with words and convert this tuple to a tuple with n grams (bi-/trigrams)
        :param tuple_with_words:
        :param n:
        :return:
        """

        # make bigrams and trigrams and store them in dictionary
        nGrams: dict[int, tuple[tuple]] = {
            2: tuple(bigrams(tuple_with_words)),
            3: tuple(trigrams(tuple_with_words))
        }

        # return a tuple that contains tuples of size n
        return nGrams[n]


    def count_ngram_occurences(string_to_check: str, language: str, n=2) -> int:
        """
        Function that takes a test string and a path to a .txt file that contains sentences from a specific language.
        The test string and the file's content are converted to bi-/trigrams. Then the occurrences of the bi-/trigrams
        in the test string in the file's bi-/trigrams are counted. These occurences are returned
        :param string_to_check:
        :param path_to_language:
        :param n: (2 equals bigram, 3 equals trigram
        :return:
        """
        # take the string to check and convert it to a tuple with bi-/trigrams
        string_to_check_grams: tuple[tuple[str]] = tuple_to_ngrams(tuple(sentence_to_list(string_to_check)), n)

        # take the path to the file that contains example strings of a specific language and convert the content
        # of this file to bi-/trigrams.
        specified_language_grams: tuple[tuple[str]] = tuple_to_ngrams(file_to_tuple(dict_paths_to_word_collections[language]), n)

        # initialize a score at 0
        score: int = 0
        for gram in string_to_check_grams:
            score += specified_language_grams.count(gram)

        # return the amount of times that a bi-/trigram from the string to check occurs in the file that contains
        # sentences that belong to a specific language
        return score

    def determine_language(string_to_test: str) -> str:
        # initializing the scores that every language
        dict_language_scores: dict[str, int] = {}

        for language in tuple(dict_paths_to_word_collections.keys()):
            dict_language_scores[language] = 0

        ''' saving language scores to 'dict_language_scores. 
        Using all available cores to speed up the process'''
        languages: tuple[str] = tuple(dict_paths_to_word_collections.keys())
        with ProcessPoolExecutor() as executor:
            # bi-/trigrams score for the dutch language
            d2: executor[int] = executor.submit(count_ngram_occurences, string_to_test, languages[0], 2)
            d3: executor[int] = executor.submit(count_ngram_occurences, string_to_test, languages[0], 3)
            # bi-/trigrams score for english language
            e2: executor[int] = executor.submit(count_ngram_occurences, string_to_test, languages[1], 2)
            e3: executor[int] = executor.submit(count_ngram_occurences, string_to_test, languages[1], 3)
            # bi-/trigrams score for german language
            g2: executor[int] = executor.submit(count_ngram_occurences, string_to_test, languages[2], 2)
            g3: executor[int] = executor.submit(count_ngram_occurences, string_to_test, languages[2], 3)
            # bi-/trigrams score for french language
            f2: executor[int] = executor.submit(count_ngram_occurences, string_to_test, languages[3], 2)
            f3: executor[int] = executor.submit(count_ngram_occurences, string_to_test, languages[3], 3)
            # bi-/trigrams score for spanish language
            s2: executor[int] = executor.submit(count_ngram_occurences, string_to_test, languages[4], 2)
            s3: executor[int] = executor.submit(count_ngram_occurences, string_to_test, languages[4], 3)
            # bi-/trigram score for italian language
            i2: executor[int] = executor.submit(count_ngram_occurences, string_to_test, languages[5], 2)
            i3: executor[int] = executor.submit(count_ngram_occurences, string_to_test, languages[5], 3)

            # saving bigram and trigram scores in dictionary
            # dutch
            dict_language_scores[languages[0]] = d2.result() + d3. result()
            # english
            dict_language_scores[languages[1]] = e2.result() + e3.result()
            # german
            dict_language_scores[languages[2]] = g2.result() + g3.result()
            # french
            dict_language_scores[languages[3]] = f2.result() + f3.result()
            # spanish
            dict_language_scores[languages[4]] = s2.result() + s3.result()
            # italian
            dict_language_scores[languages[5]] = i2.result() + i3.result()

        # return the language with the highest store
        return max(dict_language_scores, key=dict_language_scores.get)

    while True:
        sentence: str = input("enter sentence >> ")
        if sentence.lower() == 'quit':
            print('bye!')
            break
        print(determine_language(sentence))

