import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    """
    Removes stopwords from a string.
    :param text: The string to remove stopwords from.
    :return: The string with stopwords removed.
    """
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    joint_sent = " ".join(filtered_sentence)

    if len(joint_sent) == 0:
        return text
    else:
        return joint_sent
