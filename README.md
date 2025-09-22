<div align="center">
  <h1>Redes Neurais: Uma Jornada a Partir do Perceptron</h1>
  
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
  ![Status](https://img.shields.io/badge/Status-Foco_na_Teoria-blue?style=for-the-badge)
  ![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
</div>

## Introdução

Bem-vindo a este repositório dedicado ao estudo de **Redes Neurais Artificiais**! Nossa jornada começa com o bloco de construção fundamental de toda a neurocomputação: o **Perceptron**.

O objetivo deste projeto é fornecer um guia prático e teórico, utilizando a linguagem **Python** e suas poderosas bibliotecas de computação científica, para desmistificar os conceitos por trás das redes neurais. Nesta fase inicial, focaremos exclusivamente na arquitetura e na matemática que fazem o Perceptron funcionar.

---

## Arquitetura e Fórmula do Perceptron

O Perceptron é a forma mais simples de uma rede neural, modelando um único neurônio. Ele funciona como um classificador binário linear, tomando uma decisão com base em um conjunto de evidências (entradas).

A imagem abaixo ilustra sua arquitetura:
<br><div align="center">
<img width="448" height="271" alt="Image" src="https://github.com/user-attachments/assets/9a81a996-54d8-408d-be8c-beec977019cb" />
</div><br><br>


Matematicamente, o processo pode ser resumido em duas etapas:

1.  **Cálculo da Soma Ponderada (`z`):** Primeiro, calculamos a soma das entradas (`x`) multiplicadas pelos seus respectivos pesos (`w`), adicionando o bias (`b`).

```math
y = f \left( \sum_{i=1}^{n} w_i x_i + b \right)
```

3.  **Aplicação da Função de Ativação (`f`):** O resultado `z` é então passado por uma função de ativação (no caso do Perceptron clássico, uma função degrau) para produzir a saída final. Estudaremos mais adiante os tipos de funções de ativação.

---

## Componentes Chave

Para entender a fórmula, vamos detalhar cada componente:

#### 1. Entradas (Inputs - `x`)
São os dados ou características que alimentamos no modelo. Por exemplo, em um classificador de e-mail, as entradas poderiam ser a frequência de certas palavras. Cada entrada é um valor numérico.

#### 2. Pesos (Weights - `w`)
Cada entrada (`xᵢ`) é associada a um peso (`wᵢ`). O peso determina a **importância** daquela entrada na decisão final. Um peso positivo grande significa que a entrada contribui para "ativar" o neurônio, enquanto um peso negativo grande faz o contrário. **O ajuste desses pesos é o que constitui o "aprendizado"** do Perceptron.

#### 3. Bias (`b`)
O bias (ou viés) é um valor especial que não depende de nenhuma entrada. Matematicamente, ele translada a fronteira de decisão. De forma intuitiva, o bias representa o quão "fácil" é para o neurônio disparar. Um bias alto faz com que o neurônio precise de menos estímulo das entradas para ativar.

#### 4. Função de Ativação

A Função de Ativação é o componente que introduz a não-linearidade em uma rede neural, permitindo que ela aprenda padrões complexos. Ela pega a soma ponderada das entradas (`z`) e a transforma na saída final do neurônio, decidindo se e como o neurônio deve "disparar".

Abaixo estão os tipos mais comuns.

---

> #### 4.1 Função Degrau (Step Function)
> ![Função Degrau](https://img.shields.io/badge/Função_de_Ativação-Função_Degrau-808080?style=for-the-badge)

É a função de ativação mais simples, usada no Perceptron clássico. Ela produz uma saída binária.

* **Fórmula:**
    ```math
    f(z) = \begin{cases}
    1 & \text{se } z > 0 \\
    0 & \text{caso contrário}
    \end{cases}
    ```
* **Descrição:** Se a entrada (`z`) for maior que zero, a saída é 1. Caso contrário, é 0.
* **Prós:** Simples e intuitiva para problemas de classificação binária.
* **Contras:** Não é diferenciável no ponto zero, o que a torna inadequada para algoritmos de otimização baseados em gradiente, como o backpropagation.

---

> #### 4.2 Função Sigmoide (Sigmoid)
> ![Função Sigmoide](https://img.shields.io/badge/Função_de_Ativação-Função_Sigmoide-orange?style=for-the-badge)

Historicamente popular, a função Sigmoide mapeia qualquer valor de entrada para uma faixa entre 0 e 1.

* **Fórmula:**
    ```math
    \sigma(z) = \frac{1}{1 + e^{-z}}
    ```
* **Descrição:** Tem um formato de "S". Valores de entrada muito grandes são mapeados para perto de 1, e valores muito pequenos para perto de 0.
* **Uso Comum:** Frequentemente usada na camada de saída de um classificador binário, pois a saída pode ser interpretada como uma probabilidade.
* **Prós:**
    * É diferenciável, permitindo o uso do backpropagation.
    * A saída entre 0 e 1 é intuitiva.
* **Contras:**
    * **Problema do Desvanecimento do Gradiente (Vanishing Gradient):** Para entradas muito altas ou muito baixas, a inclinação da curva é quase zero. Isso pode fazer com que o aprendizado da rede seja muito lento ou pare completamente.
    * A saída não é centrada em zero, o que pode atrasar a convergência.

---

> #### 4.3 Função Tangente Hiperbólica (Tanh)
> ![Tangente Hiperbólica](https://img.shields.io/badge/Função_de_Ativação-Tangente_Hiperbólica-008080?style=for-the-badge)

A Tanh é muito semelhante à Sigmoide, mas mapeia os valores para uma faixa entre -1 e 1.

* **Fórmula:**
    ```math
    \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
    ```
* **Descrição:** Também tem formato de "S", mas é centrada em zero.
* **Prós:**
    * Por ser centrada em zero, geralmente ajuda a rede a convergir mais rapidamente do que a Sigmoide.
* **Contras:**
    * Assim como a Sigmoide, também sofre com o problema do desvanecimento do gradiente.

---

> #### 4.4 Função Unidade Linear Retificada (ReLU)
> ![ReLU](https://img.shields.io/badge/Função_de_Ativação-ReLU-brightgreen?style=for-the-badge)

A ReLU é a função de ativação mais popular e amplamente utilizada em redes neurais profundas hoje em dia, especialmente em camadas ocultas.

* **Fórmula:**
    ```math
    R(z) = \max(0, z)
    ```
* **Descrição:** É uma função muito simples. Se a entrada (`z`) for positiva, a saída é a própria entrada. Se for negativa, a saída é 0.
* **Prós:**
    * **Computacionalmente eficiente:** Muito rápida de calcular.
    * **Evita o desvanecimento do gradiente:** Para valores positivos, o gradiente é constante (igual a 1), permitindo que o aprendizado flua bem.
* **Contras:**
    * **Problema do Neurônio "Morto" (Dying ReLU):** Se um neurônio recebe uma entrada que o leva a uma saída de 0, ele pode ficar "preso" nesse estado e parar de aprender, pois o gradiente para qualquer entrada negativa é sempre zero.

---

> #### 4.5 Leaky ReLU (ReLU com Vazamento)
> ![Leaky ReLU](https://img.shields.io/badge/Função_de_Ativação-Leaky_ReLU-yellow?style=for-the-badge&logoColor=black)

É uma variação da ReLU projetada para resolver o problema do neurônio "morto".

* **Fórmula:**
    ```math
    f(z) = \begin{cases}
    z & \text{se } z > 0 \\
    \alpha z & \text{caso contrário}
    \end{cases}
    ```
* **Descrição:** Funciona como a ReLU, mas para entradas negativas, em vez de retornar 0, ela retorna `αz`, onde `α` é uma pequena constante (ex: 0.01). Isso garante que o gradiente nunca seja exatamente zero.
* **Prós:**
    * Resolve o problema do "Dying ReLU".
    * Mantém os benefícios computacionais da ReLU.

---

> #### 4.6 Função Softmax
> ![Softmax](https://img.shields.io/badge/Função_de_Ativação-Softmax-4B0082?style=for-the-badge)

A Softmax é um caso especial. Ela não é usada em camadas ocultas, mas sim na **camada de saída de classificadores multiclasse**.

* **Fórmula:**
    ```math
    S(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
    ```
* **Descrição:** A Softmax pega um vetor de `K` valores de entrada (`z`) e o transforma em uma distribuição de probabilidade. Ou seja, a saída é um vetor de `K` valores entre 0 e 1, e a soma de todos os valores é igual a 1.
* **Uso Comum:** Perfeita para problemas onde um item deve ser classificado em uma de várias categorias (ex: classificar uma imagem de animal como "cão", "gato" ou "pássaro"). A saída de cada neurônio representa a probabilidade de pertencer àquela classe.
---

## Exemplo Prático
### Prevendo Aprovação no ENEM com a Função Sigmoide

Imagine que criamos um modelo de Machine Learning muito simples, com um único neurônio, para prever a probabilidade de um estudante ser aprovado em um curso de Engenharia, que tem pesos diferentes para cada área do conhecimento.

**O Problema:** Vamos calcular a probabilidade de aprovação da estudante "Ana", com base em suas notas (escaladas de 0 a 10) e horas de estudo.

**As Entradas (Dados da Ana - `x`):**
* **Nota em Redação (`x₁`):** 8.0 (equivalente a 800)
* **Nota em Matemática (`x₂`):** 7.5 (equivalente a 750)
* **Horas de Estudo Semanais (`x₃`):** 5.0

**Os Parâmetros do Neurônio (o que ele aprendeu - `w` e `b`):**
* **Peso para Redação (`w₁`):** `0.8` (muito importante para este curso)
* **Peso para Matemática (`w₂`):** `0.7` (importante)
* **Peso para Horas de Estudo (`w₃`):** `0.2` (menos importante que a nota final)
* **Bias (`b`):** `-8` (um valor que representa a "dificuldade" base do curso)

---

### Antes do Cálculo: O Número de Euler (`e`)

Antes de aplicarmos a fórmula da Sigmoide, precisamos conhecer o seu componente mais famoso: o **Número de Euler**, representado pela letra `e`.

Assim como Pi (`π ≈ 3.14159`), `e` é uma constante matemática irracional fundamental. Ele é a base dos logaritmos naturais e aparece em todo lugar na ciência para descrever fenômenos de crescimento e decaimento.

> **Valor de Euler:** **`e ≈ 2.71828`**

Na nossa fórmula, usaremos `e` elevado a um número negativo, o que nos ajudará a "achatar" qualquer valor para um resultado entre 0 e 1.

---

### A Matemática Passo a Passo

Agora, vamos calcular a probabilidade de aprovação da Ana.

#### **Passo 1: Calcular a Soma Ponderada (`z`)**

Primeiro, combinamos todas as entradas com seus respectivos pesos e somamos o bias. Isso nos dará um único número que representa a "evidência" total para a aprovação.

* **Fórmula:**
    ```math
    z = (w₁ \cdot x₁) + (w₂ \cdot x₂) + (w₃ \cdot x₃) + b
    ```

* **Substituindo os valores:**
    ```math
    z = (0.8 \cdot 8.0) + (0.7 \cdot 7.5) + (0.2 \cdot 5.0) + (-8)
    ```

* **Resolvendo as multiplicações:**
    ```math
    z = 6.4 + 5.25 + 1.0 - 8
    ```

* **Resultado da Soma:**
    ```math
    z = 4.65
    ```
Este valor, `z = 4.65`, é a nossa "pontuação de ativação" antes de a normalizarmos para uma probabilidade.

#### **Passo 2: Aplicar a Função de Ativação Sigmoide (`σ`)**

Agora, pegamos o valor de `z` e o inserimos na função Sigmoide para transformá-lo em uma probabilidade entre 0 e 1.

* **Fórmula:**
    ```math
    \sigma(z) = \frac{1}{1 + e^{-z}}
    ```

* **Substituindo o valor de `z`:**
    ```math
    \sigma(4.65) = \frac{1}{1 + e^{-4.65}}
    ```

* **Calculando a potência de `e`:**
    Primeiro, resolvemos o termo `e⁻⁴.⁶⁵`.
    ```math
    e^{-4.65} \approx 0.00956
    ```

* **Finalizando o cálculo:**
    Agora, inserimos esse valor de volta na fórmula.
    ```math
    \sigma(4.65) = \frac{1}{1 + 0.00956} = \frac{1}{1.00956}
    ```

* **Resultado Final:**
    ```math
    \sigma(4.65) \approx 0.9905
    ```

---

### Conclusão e Interpretação

A saída do nosso neurônio foi **0.9905**.

Isso significa que, de acordo com este modelo simples, a estudante Ana tem uma **probabilidade de 99.05%** de ser aprovada no curso de Engenharia. O valor positivo e alto de `z` (4.65) indicou uma forte "evidência" a favor da aprovação, e a função Sigmoide traduziu essa evidência em uma probabilidade muito alta.


---

## Escopo Tecnológico

Este projeto irá, futuramente, utilizar as seguintes tecnologias para a implementação prática:

* **Python:** A linguagem de programação principal.
* **NumPy:** Para computação numérica e manipulação de arrays de forma eficiente.
* **Scikit-learn:** Para datasets de exemplo e métricas de avaliação.
* **Matplotlib:** Para visualização de dados e fronteiras de decisão.

---

## Próximos Passos

Com a base teórica estabelecida, os próximos passos para este repositório serão:

- [ ] **Implementação Prática em Python:** Desenvolver o código de um Perceptron do zero.
- [ ] **Perceptron Multicamadas (MLP):** Expandir o conceito para redes com múltiplas camadas.
- [ ] **O algoritmo de Backpropagation:** O mecanismo de aprendizado para redes mais complexas.
