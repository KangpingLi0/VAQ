"""COCO caption metrics used by the supplied evaluator, without the English SPICE parser."""
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


class StandardCaptionEvaluator(COCOEvalCap):
    def evaluate(self):
        image_ids = self.params['image_id']
        tokenizer = PTBTokenizer()
        references = tokenizer.tokenize({key: self.coco.imgToAnns[key] for key in image_ids})
        predictions = tokenizer.tokenize({key: self.cocoRes.imgToAnns[key] for key in image_ids})
        scorers = [(Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
                   (Meteor(), 'METEOR'), (Rouge(), 'ROUGE_L'), (Cider(), 'CIDEr')]
        for scorer, names in scorers:
            score, per_image = scorer.compute_score(references, predictions)
            if isinstance(names, list):
                for value, values, name in zip(score, per_image, names):
                    self.setEval(value, name)
                    self.setImgToEvalImgs(values, references.keys(), name)
            else:
                self.setEval(score, names)
                self.setImgToEvalImgs(per_image, references.keys(), names)
            print(names, score, flush=True)
        self.setEvalImgs()
