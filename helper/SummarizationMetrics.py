import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from transformers import BertTokenizer, BertForMaskedLM
import language_tool_python
import torch
from bert_score import score
from readability import Readability

# rouge-1 recall -> are all the words in the reference text in the generated one?
# rouge-1 precision -> how much of the generated summary is relevant?
# rouge-1 f1 -> balance between the 2 metrics

# rouge-2 -> uses bigrams instead of unigrams
# rouge-l -> treats each summary as a sequence of words then looks for the longest common subsequence (same relative order but not necessarily contiguous, eg. there are other words in the middle) -> has the advantage of not depending on consecutive ngrams matches -> captures sentence structure more accurately
# rouge-lsum -> computed over whole summary vs avg of indiv sentences

class SummarizationMetrics:
    """
    This class implements the following metrics for evaluating text summarization models:
    1. ROUGE
    2. BLEU
    3. BERTScore
    4. Readability Index
    5. Grammar Check
    
    The class takes in two arguments:
    1. reference: The reference summary
    2. generated: The generated summary
    
    Only 4 and 5 can be generated if the reference summary is not provided.
    1, 2 and 3 require both the reference and generated summaries.

    A list can also be passed in when generating only BERTScore.
    """

    def __init__(self, reference, generated):
        self.reference = reference
        self.generated = generated

    def rouge_scores(self):
        rouge = Rouge()
        scores = rouge.get_scores(self.generated, self.reference)
        return scores

    def bleu_score(self):
        reference = [nltk.word_tokenize(self.reference)]
        generated = nltk.word_tokenize(self.generated)
        score = sentence_bleu(reference, generated)
        return score

    def bert_score(self):
        # check if they are both are lists
        if isinstance(self.reference, list) and isinstance(self.generated, list):
            P, R, F1 = score(self.generated, self.reference, lang='en', verbose=True)
        else:
            P, R, F1 = score([self.generated], [self.reference], lang='en', verbose=True)
        return P, R, F1


    def readability_index(self):
        try:
            if len(self.generated) >= 100:
                r = Readability(self.generated)
                return r.flesch_kincaid()
            else:
                return "100 words required."
        except:
            return "100 words required."

    def grammar_check(self):
        tool = language_tool_python.LanguageTool('en-US')
        
        # return counts of errors
        return tool.check(self.generated)


if __name__ == "__main__":
    # Usage:
    reference_summary = "This webinar discusses the rapidly evolving field of Artificial Intelligence (AI) and its impact on various aspects of our lives. The panel of experts includes Dr. Emily Rodriguez, a renowned AI researcher and professor, Dr. James Chen, a pioneer in AI ethics, Dr. Sarah Patel, an expert in AI and its applications in healthcare, and Dr. Michael Johnson, an expert in AI and its economic implications. The panel highlights the importance of considering ethical implications, such as perpetuating biases, ensuring data privacy, and balancing innovation with societal values. AI is also revolutionizing industries and economies worldwide, but it also poses challenges in job displacement and workforce adaptation. The role of governments, businesses, and educational institutions in upskilling and retraining the workforce is crucial. The webinar emphasizes the need for vigilant and considerate consideration of ethical and societal dimensions to ensure AI remains a force for good."
    generated_summary = "Today's webinar is on the topic of Artificial Intelligence (AI). Dr. Emily Rodriguez is an AI researcher and professor. Dr. James Chen, a pioneer in AI ethics, and Dr. Sarah Patel, an expert in AI and its applications in healthcare, discuss the ethical implications of AI Artificial Intelligence has witnessed remarkable growth over the past few decades . It's now ingrained in our daily lives, from voice assistants in smartphones to self-driving cars and even in healthcare diagnostics . The future of AI holds immense promise, but it also presents important ethical and societal challenges that we need to address today and now."

    metrics = SummarizationMetrics(reference_summary, generated_summary)
    # rouge_scores = metrics.rouge_scores()
    # bleu_score = metrics.bleu_score()
    # bert_score = metrics.bert_score()
    # readability_index = metrics.readability_index()
    grammar_errors = metrics.grammar_check()

    # print("ROUGE Scores:", rouge_scores)
    # print("BLEU Score:", bleu_score)
    # print("BERT Score:", bert_score)
    # print("Readability Index:", readability_index)
    print("Grammar Errors:", grammar_errors)