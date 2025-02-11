# IndiaAI
## Hackathon: Development of a PDF reading application using LLMs

### Approch:

#### 1. PDF table extraction
  - I used pdfplumber to extract the tables in pdf and stored it in pandas dataframe.
#### 2. Embedding Rows
  - After that I embedded each row of the table and stored it in same row under "Embedding" column.
  - I used sentence_transformers library for this.
#### 3.Cosine Similarity
  - My idea was to check the cosine similarity between user's query and each rows and extract rows that had higher similarity score.
  - Used sklearn's cosine similarity function.
#### 4. Prompt Construction
  - After that I would have converted the rows to csv/string and passed it as a prompt along with the user query.
#### 5. Interface
  - I thought of using gradio for chat interface. 
  - It helps to create a web based GUI mainly for ML task.

#### Issue:
1. The prompt was not enough to generate the desired output.
2. LLM was not able to do proper calculations.
3. My approch was wrong because in some of the query I need to use "bond id" as a key to get common data from both the pdf files which was not possible in this approch.
4. Also I should have done additions of values directly in the code.

#### Alternative Approch:
1. I also though of extracting the keywords like data, political party name, company name, bond number, etc.
2. And then used it to extract info directly from dataframe.
3. After that I would have used this info as a prompt.

## Run Code
1. Create a PDF file and insert the files in it.
2. run run_pickle.py to extract data from pdf and embedd and store it in Pickle folder.
3. The pickle files are already present in the file so you can skip the above line.
3. run main.py to launch the interface.
