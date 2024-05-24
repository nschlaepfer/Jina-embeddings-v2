from sentence_transformers import SentenceTransformer, util

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

class InferlessPythonModel:
    def initialize(self):
        self.model = SentenceTransformer("jinaai/jina-reranker-v1-tiny-en",trust_remote_code=True)
        # control your input sequence length up to 8192
        self.model.max_seq_length = 1024

    def infer(self, inputs):
        sentences = inputs["sentences"]
        embeddings = self.model.encode(sentences)
        return {"result": embeddings}
    def finalize(self, args):
        self.pipe = None
