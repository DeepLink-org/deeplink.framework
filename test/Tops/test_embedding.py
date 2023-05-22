import torch


class EmbeddingTest(torch.nn.Module):
    def forward(self, weight, indices):
        return torch.ops.aten.embedding.default(weight, indices)


if __name__ == "__main__":
    weight = torch.randn(10000, 4096)
    indices = torch.randint(10000, (1, 12))
    embedding_test_module = EmbeddingTest()
    compiled_module = torch.compile(embedding_test_module, backend="topsgraph")

    long_star = '*' * 20
    print(f"{long_star} test embedding start {long_star}")
    test_result = compiled_module(weight, indices)
    print(f"{long_star} test embedding end {long_star}")
    print(f"{long_star} reference embedding start {long_star}")
    ref_result = embedding_test_module(weight, indices)
    print(f"{long_star} reference embedding end {long_star}")
    print(f"{long_star} result compare {long_star}")
    print(f"torch.allclose(test, ref): {torch.allclose(test_result, ref_result, equal_nan=True)}")
    print(f"torch.all(torch.eq(test, ref)): {torch.all(torch.eq(test_result, ref_result))}")
