# LLM-twin-from-scratch

Reference repository: https://github.com/PacktPublishing/LLM-Engineers-Handbook/tree/main

My Version of the LLM Engineers Handbook. I chose to make my own instead of forking because I wanted to code all of the files from scratch to gain a more in depth understanding of the underlying processes for the project.

One major thing I learned in grad school is that to truly understand something you have to do it, not just read about it. While I have read through most of the LLM Engineers Handbook already, I felt that I was lacking in understanding of some of the files and how they operate within the FTI pipeline framework. That is why I set up this repository. While many of the files I will be coding here will be very similar or at times exactly like the files from the LLM Engineers Handbook Repo, I will be making modifications when necessary to fit my needs while extensively documenting the process.

The goal is to create my own LLM Twin chatbot and deploy it as an example on the Huggingface platform so that others can view my work and as a proof of concept to future employers. The LLM Twin creation process will broaden my knowledge base of how to fully enact a model of my own that can be scaled to production. It is important to note that I have limited resources for future scaling so this project may never get that far, but the goal is not to use this model personally. I want to learn and broaden my knowledge. Let's get started.

---

### Day 1:

For the first day of me copying the files I chose to begin by setting up the `pyproject.toml` file and the docker-composition file. I did this because at the beginning of the book the pyproject file sets the precedent for the entire file system by setting up the necessary package installations and dependencies. I made some edits to the file to set up my own personal etl pipeline instead of the one from the original authors. Based on this, I set out to create the file system and dependencies to create the etl pipeline for my own personal data.

---

### Day 2:

I began exploring the `crawl_links.py` file and followed the file lineage to really understand where each file pulled its functions and classes from. Based on my learning I began crafting each python file and documenting my learning of each class and function. Since the `crawl_links.py` file is designed to crawl different webpages specified in the authors `etl_configurations.yaml` files, I chose to begin creating the `llm_engineering` framework files needed to make mongodb connections and create the necessary crawler dispatchers for each document class.

---

### Day 3:

Today is a continuation of yesterday with the creation and typing of the file system. Already I have begun to understand how the crawler uses the different document classes to create and access mongodb for NoSQL documents. I have written many files that perform various functions for the etl pipeline. Today my plan is to keep writing with the goal of finishing the `digital_data_etl.py` file and all of its dependencies. Over the next couple of days when I finish these files I plan on attempting to run the pipeline from the book but with my own data. I should be able to get the mongo db connection and the docker containers up and running without using the authors full file set, hopefully. From there I will continue building each pipeline and building my own personal LLM Twin.

---

#### Important Notes:

- Based on todays creation of the first pipeline being used via Zenml I want to make my knowledge more concrete by writing out my thoughts and understanding.
- The authors have created the file system intricately such that we can use any orchestrator to execute the knowledge necessary to run the pipelines.
- **What does this mean?**
  - It means that all of the `@pipeline` steps and `@step` portions of the pipeline are stored separately from the pipelines module in the `llm_engineering` file folder.
  - This allows the user to easily switch orchestrators if necessary since all of the application and domain logic is stored in `llm_engineering` and the zenml orchestrator logic is stored in the pipelines and steps folders. So, to make the switch I would only need to change the zenml code, not the full logic.
  - Within the steps and pipeline module, they only used the things needed from the `llm_engineering` module, this keeps the logic separate.
- Values that are to be returned from zenml have to be serializable, meaning that it can be converted to a format suitable for storage or transmission and later reconstructed (deserialized) into its original form.
- All data and artifacts along with their metadata are stored in the zenml dashboard and can be viewed there.
- Metadata is added manually to the artifact allowing me to precompute and attach anything that is helpful for dataset discovery across projects.

---

#### `run.py`:

- Everything for zenml can be run via this file.
- Given that this file holds and entails all of the pipelines and commands within the code, I want to save writing this file until I have fully finished the book and all of the logic, pipelines, etc.

---

#### MongoDB:

MongoDB is chosen as the storage location for the etl pipeline because not many documents are used in the proof of concept. Thus small scale statistics can be calculated and little cost incurred. Should the twin be scaled out to include millions of documents or more then a large scale data warehouse such as Snowflake or BigQuery should be used.

---

#### Crawlers:

Each of the crawler classes implements custom logic to access the articles, posts, and repositories with the exception of the `CustomArticleCrawler`:

- **GithubCrawler** - Crawls and stores Github repos.
- **MediumCrawler** - Crawls and stores Medium articles based on HTML.
- **LinkedInCrawler** - Crawls and stores LinkedIn posts.
- **CustomArticleCrawler** - Crawls articles outside of the designated domains for the other crawlers. (No custom logic - primarily a fallback.)

---

#### Decisions:

I coded each of these crawlers, later to find out that based on the sources I will be using for my data, I will not be using the `CustomArticleCrawler` or the `MediumCrawler`.

- The `CustomArticleCrawler` uses LangChain packages as a fallback method to crawl different domains and is generally not used in production environments anyway.

#### ORM:
 ORM is a technique that lets you query and manipulate data from a db. Instead of writing SQL or API queries all of the complexity if captured by an ORM class that handles all of the underlyring database operations. This removes the manual coding of database operations and reduces the need to write all of the underlying code needed to perform them.

 ORMs interact with SQL databases such as PostgreSQL or MySQL. 

#### ODM:

 ODM works similarly to ORM but instead of connecting to SQL databases it connects to NoSQL databases. In this project I am working with unstructured data so the data structure is centered on collections. These collections store JSON like documents rather than rows and columns in tables. It simplifies working with NoSQL databases and maps object-oriented code to JSON like documents. This is the type of module that is implemented in the nosql.py file.
 
 The class in this file is called NoSQLBaseDocument and is used as the base class to store all of the objects brought in by each of the crawlers.

#### Conclusions:
 By using the ODM class and its stored settings for each document in coordination with the zenml artifacts I can more modularly debug my code, monitor and trace the results and metadata for each pipeline. 
 
---

 ### Day 4:

 Today I continued reading back through chapter 4 of the LLM Engineers handbook. I had already read through the chapter briefly but today I went more in depth. In fact, since I am coding out the entire repo, I decided to start creating all of the dependencies for the feature engineering pipeline. The feature engineering pipeline encompasses all 5 portions of the RAG pipeline. So far I have been able to manage going through just the cleaning portion of the pipeline. 

--- 
 #### SingletonMeta:

 I made many classes including the metaclasses used to ensure consistency when making connections via Qdrant on multithreaded systems. This was something entirely new I learned about network connections. I didn't know what a lock object or a metaclass was until today. Apparently lock objects, prevent multiple instances of the same class from being created before the first established connection is entirely finished with the process. As I mentioned before, this metaclass, `SingletonMeta`, represents the base class for all vector based storage in the Qdrant database, because all other subclasses involved in the cleaning step inherit from this subclass. Also, all of the cleaning, chunking and dispatching involves intricate connections between these subclasses. In short, without establishing this connection, any of my instances could become corrupted due to parallel creation. 

---
 
 ### Day 5 and 6:

Today I finished hardcoding the full `feature_engineering.py` pipeline. This pipeline is broken down into four primary .py files. I will go into a bit of detail about them and how they all work together to push the features from the extracted documents in Mongo DB into Qdrant via a Zenml Orchestrator pipeline.

---

#### `query_data_warehouse.py`

This file serves as a Zenml step to fetch all of the raw data for the user from the data warehouse (Mongo DB). It is designed to take each of my document classes and fetch all of the data for those classes. My data will consist of LinkedIn posts and GitHub repositories only but there is functionality included to pull Articles from Medium and other domains as well. In this step, the data is pulled in parallel processes to increase efficiency. Alongside the actual content from each document, the metadata is efficiently stored in dictionaries for monitoring and use in the Zenml dashboard.

---

#### `clean.py`

This file serves as a Zenml step to take the queried documents from the `query_data_warehouse.py` step and clean for further transforming and processing. Here we are iterating through all of the documents and delegating all of the cleaning logic to subclasses of a created CleaningDispatcher class. Each subclass is designed to take and clean documents for each of their respective categories. Here the metadata for these cleaned documents is also stored for monitoring. 

---

#### `rag.py`
This file serves as the next Zenml step to take the cleaned documents and chunk and embed them based on an overlap procedure. Overlapping allows for faster searching when stored in a vector db becuase of pieces of the same text being found in multiple chunks. The chunk settings here will be experimented with when I run the full project to improve the results of my LLM-Twin responses. The authors used a chunk size of 500 and an overlap of 50, but since my dataset is smaller I may opt to go with smaller sizes and overlap parameters. Each chunk also contains its chunk metadata and embedding metadata. 

---

#### `load_to_vector_db.py`

This file serves as the final Zenml step in the pipeline. Its function is to take the cleaned, embedded, and chunked documents and load them into our selected vector database. Qdrant is the selected vector database for this project, and since I have never used a vector database, I don't really have much of a preference for which is best, more on Qdrant below.

The way this step works is based on using different dispatchers based on the collection each article, post or code repository is stored in. All documents must be grouped by their data category and then loaded in bulk to the Qdrant DB. For this the authors introduced Pydantic, which is the go-to Python package for writing data structures with out-of-the-box type validation. In short, pydantic allows me to use Domain Driven Design principles to construct the correct hiearchy of the domain I am working with. Simply put, I can divide this step into two different domains. 
  - The data category: Article, Post, Repo
  - The data state: Cleaned, Embedded, Chunked

Each state has a base class that inherits from a VectorBaseDocument class. They also inherit from the abstract base class (ABC) which means objects cannot be initialized out of these classes, they can only inherit from them. 

For each state their are individual subclasses for the different data categories. The authors gave a reference image for this but I will draw one of my own to imitate and ensure my understanding. 

![image](https://github.com/user-attachments/assets/495d652e-c29d-4a74-af0f-b004298d385a)

 


    

        

        






Within each of the subclasses in for the different states there are internal Config classes that point to the given settings for each type of document. This allows me to accurately define each document after storing in the vector db. Also, an enum is defined to aggregate all of the data categories into a single structure.

---

#### OVM class (VectorBaseDocument)
This is the OVM base class that will be used to structure a single record's attributes from the vector database. It is initialized as type UUID4 as the unique identifyer and inherits from Pydantics BaseModel class, ABC, and pythons Generic[T] class. This means that all subclasses will adapt, and inherit the settings of this class. If you are curious and want to see more about the structure of this class, please go look at the code as I have documented each of their functions and usage.

---

#### Qdrant 
The authors chose this vector database as the one to be implemented for the LLM-twin because it is "one of the most popular, robust and feature-rich databases." It was chosen because of how light it is and has a standing in the industry that sets a precedent that it will be around for many years to come. Qdrant is also used by many of the big players in the industry such as Disney, Microsoft, and Discord. It offers the best trade off between RPS (Requests per Second), latency, and index time. This allows querying and indexing of vectors to be quick and efficient for the LLM twin training and generation process.

---

#### Conclusions:

The Rag feature pipeline serves as a quick and efficient method of cleaning, chunking, embedding and storing the features into Qdrant in a way that is modular and follows MLOps best practices and principles. It will allow me to make changes in the code without having to modify then entire code base because all of the classes are wrapped and work together to support a generic design. In future projects when I want to create a production ready llm, these principles and coding structures will allow me to make simple and efficient modifications.

Next I will begin learning about how to create custom datasets to train my llm-twin.

---

### Day 7:
#### Supervised Finetuning
Supervised finetuning is way to take a pre-trained existing llm and finetune it for a potential use case. In this case I want to be able to instruction tune my model on question answer pairs to improve the accuracy of responses from my llm-twin. For Chapter 5, I will be learning how to create a high quality instruction dataset, implement SFT techniques and implement fine-tuning in practice. I have implemented supervised fine-tuning techniques before for text-classification tasks but I have never instruction-tuned a model using the question answer pairs. This will be a great learning experience.

---

#### Creating an instruction tuned dataset
This is one of the most difficult parts of the fine-tuning process becuase I will need to transform my text into a format that supports both instructions and answers. Data quality is crucial. So there needs to be extensive modifications and monitoring to ensure this quality is obtained. 

Quick Bullets about the general framework of creating the dataset.
  - Can be trained on both instructions and answers or answers only. 
  - Instructions can be broken into multiple fields (System, Instruction and Output)
    - System provides metadata context to steer the llm in a general direction.
    - Instruction provides the necessary data and a task.
    - The Output shows the expected answer from the given inputs.
  - Data quality can be divided into three primary domains.
    - Accuracy: Factual correctness
    - Diversity: Encompasses a wide range of use cases
    - Complexity: Should include multi-step reasoning problems to push boundaries of the model.
  - Smaller models require more high-quality samples, larger models require less.
  - Good fine-tunes require around 1 million samples.
  - For specialized fine tuning less is required.
  - ##### Task Specific Models:
    - Designed to excel at one particular function (translation, summarization, etc)
    - Efficient performance even with smaller models.
    - Anywhere between 100 and 100,000 samples.
  - ##### Domain Specific Models:
    - Aimed to fix the model towards more specialized linguistics in a particular field.
    - Sample size depends on the breadth of the technical corpora.
    - Data Curation can be more challenging.
  - Few shot prompting can be a viable alternative to fine-tuning, it depends on the use case.
  - ##### Rule Based filtering:
    - Relies on explicit, pre-defined rules to evaluate and filter data samples.
    - Length filtering sets thresholds for the acceptable length of responses in the dataset.
      - Extremely short responses often lack the neccessary information needed.
      - Extremely long responses may contain irrelevant or redundant information.
      - The maximum length is determined based on the use case. If we need technical sumaries we may opt for longer responses, while shorter ones may be needed for general summaries.
    - Keyword exclusion focuses on the sample content rather than structure.
      - Creates a list of keywords or phrases for low-quality content and then filters out samples that contain them.
    - Format checking ensures that all of the samples meet a designated format. (Important for code checking etc.)
  - ##### Data Deduplication
    - Duplicates can lead to:
      - Overfitting - Memorization
      - Biased Performance - Skewing
      - Inefficient training - Increased training time
      - Inflated evaluation metrics - Overly optimistic performance.
    - Finding and removing exact duplicates can be done using MD5, of SHA-256 hashing algorithms
    - Normalization of text can help with finding duplicates as well.
    - Fuzzy deduplication can be done best through MinHash.
        - Generates compact representations for the data.
        - Help capture the essence of the data while reducing dimensionality.
        - Transforms data into shingles, applies hashes on those and selects min hash values to form signature vectors.
        - Vectors can be compared via Jaccardian similarity to measure overlap.
    - Semantic similarity focuses on the meaning of text for deduplication.
      - Converts words and entire samples to vector representations where Word2Vec, Glove, and FastText can be used to find semantic relationships.
    - BERT and other LLMs can also be used to find the similarity of word or document vectors as well.
      - Cosine (angular distance) and Euclidean similarity used here. 
      -  Clusters can also be formed for Vectors via K-means, DBSCAN or hierarchical clustering.
  - ##### Data Decontamination
    - This is done by ensuring that there is no overlap between training and evaluation anf test sets.
    - Can be done through exact matching (hashing or string comparisons).
    - Can also be done via similarity methods or MinHashing.
    - This can be done easily by performing proper deduplication.
    - Automation of this process is ideal in production environments.
  - ##### Data Quality Evaluation:
    - Human quality assessments can be used but consume resources.
    - LLMs can be used as a judge by including different prompt domains to rate the quality.
      - There is inherent bias here because the LLM's generally favor the first answer.
      - They have intra model favoritism, favoring models from the same family of models.
        - Multiple models can be used to address this.
      - For chatbots we want the human ratings and LLM ratings to agree on the quality (80% of the time).
    - Reward models can also be used giving a score as an output for the quality.
      - Takes multiple scores from specific dimensions, refer to ArmoRM architechture.
  - ##### Data Exploration
    - Manual Exploration can be time-consuming but is still important.
      - Reveals errors that can occur from automated proceses.
    - Statistical analysis can reveal insights into the vocabulary density, biases and correct representation.
    - Clustering based on topics can also be used to examine which topics are the mentioned most often and reveal biases and subtopics within different coding languages.
  - ##### Data Generation
    - When public data is unavailable this can be critical, this is particularly so in more specialized applications.
    - Generation begins with the preparation of prompts that serve as the foundation for generating new examples.
    - Generally the prompts include specific instructions, examples, and constraints to ensure generated data is in alignment with the desired content and format.
    - Generated data needs to be controled for inherent biases and errors that come from the underlying model.
  - ##### Data augmentation
    - Process of increasing the qunatity and quality of existing data samples.
    - Evol-Instruct method
      - Uses LLMs to evolve more simples instructions into more qualitative ones.
      - There are 2 main strategies, in-depth and in-breadth evolving.
      - In depths focus is on enhancing complexity by introducing additional requirements for instructions, asking deeper questions, replacing general concepts with more specific ones, introducing multi-step reasoning, and adding more complex data (XML, JSON, code).
      - In breadth generates entirely new instructions inspired by those that already exist and createing more rare or longer examples for the same domain.
    - Ultra feedback method is used by focuing more on answer quality instead of the instruction quality.
      - It uses a more advanced model to evaluate the answers and provide critiques and scores for the answers.
  - #### Final Remarks
    - For my datasets I will be implementing a pipeline that does the following:
      - Takes my raw-text documents
      - Cleans and Chunks them
      - Takes the chunks and uses GPT 4o mini to generate instruction answer pairs
      - Filters these pairs based on pre-defined rules to create the final instruction dataset.
    - My aim is to finish this step based on the authors templates and then refine after I do my own evaluation when I deploy the project.

---

### Day 8 and 9:
#### Finishing the dataset generation pipeline `generate_datasets.py`
On day 8 I noticed that the generation pipeline was different than the baseline one mentioned in the book. The generation pipeline encompasses not only the creation of Instruction datasets, but also preference datasets. I decided to go ahead and code out the entire process. The preference dataset is very similar to the instruction dataset generation but there was a strong difference in the prompt engineering used to ensure correctness of the outputs. The pipeline consists of 5 different zenml steps from 5 seperate python files, each with its own purpose and functionality. I will be going a bit more in depth about how each of them work and explaining my understanding of the underlying llm engineering logic used to incorporate them. Since I have already gone in depth about the Instruction dataset generation that is in in the `generate_instruction_dataset.py` file,  I will be skipping that file and focusing on the other ones.

---
#### `query_feature_store`
This file is designed to create a zenml step that queries the documents in mongo db based on the NoSQLBaseDocument class. It pulls all of the document types using the ThreadPoolExecutor and uses other functions such as bulk_find from the CleanedDocument class.
#### `create_prompts.py`
This creates a zenml step that is built upon multiple other classes and files. It's purpose is to intake queried clean documents after calling the query_feature_store() function and generate a list of generated dataset samples that are built on top of engineered prompts. This allows for the instruction and preference datasets to be created because they feed off of prompts to ensure a set JSON format as the output. 
#### `generate_instruction_dataset.py`
This file creates a zenml step that is designed to take in a set of generated prompts and create an instruction dataset that follows a specific template string pattern. The prompts are split based on a test_split_size and stored separately. The instruction dataset output is designed to be a set of instruction-answer pairs that are consistent, non-redundant, and are output in JSON format. This allows for their proper storage in huggingface and parsing when it comes to training the model for output generation. 

Format:
    prompt_template_str = """Based on the following extract, generate five instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context. Each answer \
must provide a relevant paragraph based on the information found in the \
context. Only use concepts from the context to generate the instructions. \
Instructions must never explicitly mention a context, a system, a course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \

Example instruction: Explain the concept of an LLM Twin. \
Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice. \
It's designed to write just like you by incorporating these elements into a language model. \
The idea is to create a digital replica of your writing habits using advanced AI techniques. \

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {"instruction": "...", "answer": "..."}, 
    ...
]

Extract:
{extract}

#### `generate_preference_dataset.py`
This file is creates a zenml step for the pipeline to generate preference datasets. It also takes in a set of prompts and follows a very similar pattern as the `generate_instruction_dataset.py` file. It splits the prompts based on a test_split_size and stored separately. These datasets also are created from a prompt template that asks the model to store them in a specific JSON format. However, their format is different as they create output based on instruction-answer triples.

Format:
    prompt_template_str = """Based on the following extract, generate five instruction-answer triples. Each triple should consist of:
1. An instruction asking about a specific topic in the context.
2. A generated answer that attempts to answer the instruction based on the context, named as 'rejected'.
3. An extracted answer that is a relevant excerpt directly from the given context, named as 'chosen'.

Instructions must be self-contained and general, without explicitly mentioning a context, system, course, or extract.

Important:
- Ensure that the extracted answer, the chosen one, is a verbatim copy from the context, including all punctuation and apostrophes.
- Do not add any ellipsis (...) or [...] to indicate skipped text in the extracted answer.
- If the relevant text is not continuous, use two separate sentences from the context instead of skipping text.

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {
        "instruction": "...", 
        "rejected": "...", 
        "chosen": "..."
    },
    ...
]

Extract:
{extract}

#### `push_to_huggingface.py`
This file creates a zenml step that serves as the final step in the pipeline. Essentially it ensures that the proper access token is available in the users settings file, takes the generated datasets from the other steps and pushes them to the Huggingface hub along with their metadata.
