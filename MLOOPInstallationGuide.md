## Instalação

Instruções gerais podem ser encontradas em https://m-loop.readthedocs.io/en/stable/install.html. Recomenda-se a instalação direto da fonte em um ambiente do Anaconda. Para preparar um ambiente novo para a utilização do M-LOOP, siga os seguintes passos:

1. Instale o Anaconda. Se ele já tiver instalado, verifique que está atualizado executando:

conda update conda

2. Crie um novo ambiente utilizando 

conda create -n myenv

substituindo "myenv" pelo nome do ambiente.

3. Ative o novo ambiente com a linha

conda activate myenv

4. Para instalação do M-LOOP da fonte, é necessário o pacote setuptools. Para instalá-lo execute

conda install setuptools

5. Na pasta onde deseja fazer o download dos arquivos do repositório do pacote, execute 

git clone git://github.com/michaelhush/M-LOOP.git
cd ./M-LOOP
python setup.py develop

A primeira linha fará uma cópia do repositório, a segunda muda para o diretório do M-LOOP, e a última executa a instalação do pacote.

6. O script se encarregará de instalar todos os pacotes dos quais o M-LOOP faz uso, mas estes pacotes dependem de outros que o ambiente novo ainda não terá. Para instalar as últimas dependências, execute

conda install matplotlib
conda install scikit-learn
conda install tensorflow

Vários pacotes novos serão instalados ou atualizados.

7. Para verificar sua instalação, execute, ainda na pasta do M-LOOP

python setupy.py test

Avisos são normais e são consequência da escrita do pacote. No caso de erros, entre em contato. A lista de pacotes instalados no ambiente pode ser exibida executando

conda list

