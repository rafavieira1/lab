# Lab - Algoritmos de Aprendizado

Laboratório de implementação de algoritmos de busca e aprendizado.

## Como executar

### Pré-requisitos
- Python 3.10 ou superior
- Poetry (gerenciador de dependências)

### Instalação

1. **Instalar Poetry** (se não tiver):
```bash
pip install poetry
```

2. **Instalar dependências**:
```bash
poetry install
```

3. **Executar o exemplo principal**:
```bash
poetry run python examples/teste.py
```

### O que o projeto faz

O arquivo `examples/teste.py` demonstra:
- Autoencoder em dados MNIST
- Visualização 2D com PCA e t-SNE
- Treinamento de rede neural com PyTorch

### Estrutura do projeto

```
lab/
├── examples/
│   └── teste.py          # Exemplo principal
├── src/
│   └── lab/
│       ├── dataset.py    # Carregamento de dados
│       └── perceptron.py # Implementação do perceptron
└── pyproject.toml        # Configuração do Poetry
```
