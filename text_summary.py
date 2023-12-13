import spacy 
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

text = """BTS (Korean: 방탄소년단; RR: Bangtan Sonyeondan; lit. Bulletproof Boy Scouts), also known as the Bangtan Boys, is a South Korean boy band formed in 2010. The band consists of Jin, Suga, J-Hope, RM, Jimin, V, and Jungkook, who co-write or co-produce much of their material. Originally a hip hop group, they expanded their musical style to incorporate a wide range of genres, while their lyrics have focused on subjects including mental health, the troubles of school-age youth and coming of age, loss, the journey towards self-love, individualism, and the consequences of fame and recognition. Their discography and adjacent work has also referenced literature, philosophy and psychology, and includes an alternate universe storyline.

BTS debuted in 2013 under Big Hit Entertainment with the single album 2 Cool 4 Skool. BTS released their first Korean and Japanese-language studio albums, Dark & Wild and Wake Up respectively, in 2014. The group's second Korean studio album, Wings (2016), was their first to sell one million copies in South Korea. By 2017, BTS had crossed into the global music market and led the Korean Wave into the United States, becoming the first Korean ensemble to receive a Gold certification from the Recording Industry Association of America (RIAA) for their single "Mic Drop", as well as the first act from South Korea to top the Billboard 200 with their studio album Love Yourself: Tear (2018). In 2020, BTS became one of the few groups since the Beatles (in 1966–1968) to chart four US number-one albums in less than two years, with Love Yourself: Answer (2018) becoming the first Korean album certified Platinum by the RIAA; in the same year, they also became the first all-South Korean act to reach number one on both the Billboard Hot 100 and Billboard Global 200 with their Grammy-nominated single "Dynamite". Follow-up releases "Savage Love", "Life Goes On", "Butter", and "Permission to Dance" made them the fastest act to earn four US number-one singles since Justin Timberlake in 2006.

As of 2023, BTS is the best-selling artist in South Korean history according to the Circle Chart, having sold in excess of 40 million albums.[2] Their studio album Map of the Soul: 7 (2020) is the third best-selling album of all time in South Korea, as well as the first in the country to surpass both four and five million registered sales. They are the first non-English-speaking and Asian act to sell out concerts at Wembley Stadium and the Rose Bowl (Love Yourself World Tour, 2019), and were named the International Federation of the Phonographic Industry's (IFPI) Global Recording Artist of the Year for both 2020 and 2021. The group's accolades include multiple American Music Awards, Billboard Music Awards, Golden Disc Awards, and nominations for five Grammy Awards. Outside of music, they have addressed three sessions of the United Nations General Assembly and partnered with UNICEF in 2017 to establish the Love Myself anti-violence campaign. Featured on Time's international cover as "Next Generation Leaders" and dubbed the "Princes of Pop", BTS has also appeared on Time's lists of the 25 most influential people on the internet (2017–2019) and the 100 most influential people in the world (2019), and in 2018 became the youngest recipients of the South Korean Order of Cultural Merit for their contributions in spreading the Korean culture and language.

On June 14, 2022, the group announced a scheduled pause in group activities to enable the members to complete their mandatory South Korean military service, with a reunion planned for 2025. Jin, the oldest member, enlisted on December 13, 2022, followed by J-Hope on April 18, 2023, and Suga on September 22, 2023."""

def summarizer(rawdocs, percent):
    stopwords = list(STOP_WORDS)
    # print(stopwords)

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdocs)
    # print(doc)

    tokens = [token.text for token in doc]
    # print(tokens)

    word_freq = {}
    for word in doc:
        if(word.text.lower() not in stopwords and word.text.lower() not in punctuation):
            if( word.text not in word_freq.keys()):
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1
                
    # print(word_freq)

    max_freq = max(word_freq.values())
    # print(max_freq)

    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq
        
    # print(word_freq)

    sent_token = [sent for sent in doc.sents]
    # print(sent_token)

    sent_scores = {}
    for sent in sent_token:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]
                    
    # print(sent_scores)

    select_len = int(len(sent_token) * (int(percent)/100))
    # print(select_len)

    summary = nlargest(select_len, sent_scores, key = sent_scores.get)
    # print(summary)

    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    # print(summary)
    # print("Length of original text",len(text.split(' ')))
    # print("Length of sumary text",len(summary.split(' ')))
    
    return summary, doc, len(rawdocs.split(' ')), len(summary.split(' ')) 