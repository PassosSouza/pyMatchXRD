# PyMatchXRD  

**Ferramenta computacional para análise de padrões de XRD e identificação de estruturas cristalinas**  

Este repositório contém os códigos desenvolvidos para meu trabalho final na disciplina **Física Computacional**, ministrada pelo professor **João Nuno Barbosa**. O objetivo foi desenvolver modelos para identificar **células unitárias** e **estruturas cristalinas** a partir de padrões experimentais de **difração de raios-X (XRD)** usando algoritmos genéticos e outras técnicas de otimização.

---

##  Estrutura do Projeto  

### `pyXRD`  
Módulo principal para simulação de padrões de XRD a partir de estruturas cristalinas conhecidas.  
- **Exemplo de uso**: [`Usar_pyXRD.ipynb`](https://github.com/PassosSouza/pyMatchXRD/blob/main/Usar_pyXRD.ipynb)  
- **Código-fonte**: [`pyXRD/`](https://github.com/PassosSouza/pyMatchXRD/tree/main/pyXRD)  

---

##  Algoritmos Principais  

### 1. **Algoritmo Genético para Identificação de Célula Unitária**  
- **Notebook**: [`GA_Cell_Definitivo.ipynb`](https://github.com/PassosSouza/pyMatchXRD/blob/main/GA_Cell_Definitivo.ipynb)  
- **Objetivo**: Otimiza parâmetros de rede (`a, b, c, α, β, γ`) para corresponder a padrões experimentais de XRD.  

### 2. **Algoritmo Genético para Refinamento de Posições Atômicas**  
- **Notebook**: [`GA_Struct_Compare.ipynb`](https://github.com/PassosSouza/pyMatchXRD/blob/main/GA_Struct_Compare.ipynb)  
- **Versão Aprimorada**: [`GA_Struct_Definitv.ipynb`](https://github.com/PassosSouza/pyMatchXRD/blob/main/GA_Struct_Definitv.ipynb)  
  - **Nova Funcionalidade**: Permite fixar parâmetros específicos (ex: constantes de rede) para direcionar a otimização para estruturas esperadas.  

---

##  Outros Métodos Testados  
Explore abordagens alternativas discutidas no relatório do projeto:  
- [`Outros Metodos/`](https://github.com/PassosSouza/pyMatchXRD/tree/main/Outros%20Metodos)  

---

##  Instalação & Dependências  
Para executar os códigos:  
Clone o repositório e instalar dependencias:  
   ```bash
   git clone https://github.com/PassosSouza/pyMatchXRD.git
   cd pyMatchXRD
   pip install numpy matplotlib pymatgen numba
   ```

---

## Relatório

O relatório completo está disponícel no arquivo [`FisComp_pyMatchXRD.pdf`](https://github.com/PassosSouza/pyMatchXRD/blob/main/FisComp_pyMatchXRD.pdf), no qual são apresentadas as ideias propostas e como foram aplicadas os métodos.

---
Sinta-se à vontade para explorar, adaptar e contribuir com sugestões ou melhorias!
