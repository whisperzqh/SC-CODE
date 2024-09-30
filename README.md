## On the Applicability of Code Language Models to Scientific Computing Programs

**The data can be downloaded through the figshare link:** https://figshare.com/s/46ec7f0cff1c6072abc6

## Benchmark Information

+ ```./raw_data``` contains all the code files collected from GitHub repositories and the documentation collected from the official documentation of the three languages.

  + ```./raw_data/github_data``` contains all the source code files from the collected GitHub repos.

    |            | Julia | MATLAB |  R   |
    | ---------- | :---: | :----: | :--: |
    | # of repos |  619  |  506   | 542  |

    + data format: Code corpus are saved in json format files. 

      ```
      $lang_data.json
      {
          path: the source file path
          code_string: code string in the file
      }
      ```
    
  
  
    + ```./raw_data/documentation``` contains the documentation information, including the description, API name, signature, parameters, usage examples (if has), etc.
  
      |                    | Julia | MATLAB |   R   |
      | ------------------ | :---: | :----: | :---: |
      | # of documentation | 2,211 | 4,142  | 6,851 |
  


+ ```./SC-CODE``` contains the extracted NL-code pairs from the source code files that can support code generation and code summarization. It also contains the API doc information from the documentation that can serve as the knowledge base.

  + ```./SC-CODE/NL_code``` 

    + We extract the method and comment pairs based on heuristic rules, and following FSE'22 paper: [Are We Building on the Rock? On the Importance of Data Preprocessing for Code Summarization](https://github.com/BuiltOntheRock/FSE22_BuiltOntheRock) to clean the dataset.
    + data format: Method and NL are saved in jsonl format files.

      ```
      $lang/$split.jsonl
      {
          path: the path to the original file
          language: the programming language
          code: the code refers to function in the original file
          code_tokens: tokenized version of `code`
          docstring: the comment or docstring
          docstring_tokens: tokenized version of `docstring`
          partition: train/val/test
      }
      ```

  + ```./SC-CODE/code_doc```

    + data format: API and description information are saved in json format files.

      ```
      $lang_doc.json
      {
          name: API name
          code_string: API usage example code
          des_string: description of the API
      }
      ```

  + All the data extraction and pre-processing code is available at: ```./SC-CODE/DataPreprocessing/```

+ ```./SC-API-CODE``` contains the NL-code pairs that include rich API calls related to computational mathematics based on SC-CODE. It also contains the API doc information that belong to libraries related to mathematical calculations and graphics.

    + ```./SC-API-CODE/NL_code``` 

      + We extracted the API calling statements in the code using [tree-sitter](https://tree-sitter.github.io/tree-sitter/) and saved the detailed information.
      + data format: Method and NL are saved in jsonl format files.
    
        ```
        $lang.jsonl
        {
            path: the path to the original file
            language: the programming language
            code: the code refers to function in the original file
            docstring: the comment or docstring
            partition: train/val/test in SC-CODE
            function_call: detailed information about the extracted API calls
            line_with_function_call: number of lines containing API calls
            function_call_in_doc: detailed information about the extracted API calls that can be found in the corresponding documentation
            num_of_func_call_in_doc: number of lines containing API calls that can be found in the corresponding documentation
        }
        ```

    + ```./SC-API-CODE/code_doc```
    
      + data format: API and description information are saved in jsonl format files.
    
        ```
        $lang_doc.jsonl
        {
            name: API name
            code_string: API usage example code
            des_string: description of the API
            category: libriary name
        }
        ```

- ```./data_for_semantic_correctness``` contains the high-quality computation-related NL-code pairs with test cases for semantic correctness evaluation of code generation. It also contains the evaluation script.

  - ```./data_for_semantic_correctness/testcases_data``` 

    - We meticulously generated two test cases for each piece of data for the semantic correctness evaluation.

    - data format: NL-code pair and test cases are saved in jsonl format files.

      ```
      $lang.jsonl
      {
          task_id: unique ID
          prompt: natural language comment and function signature, used as prompt during evaluation
          entry_point: function name
          reference_solution: ground truth
          test: test cases
      }
      ```

  - The automated evaluation script is available at: ```./data_for_semantic_correctness/evaluation.py```. Its use requires ensuring that the operating environment of Julia, MATLAB, and R is installed.

## Model Fine-tuning


### Code Summarization 

+ Fine-tuning CodeBERT

  Following CodeXGLEU, we fine-tune CodeBERT on our code summarization dataset using their [provided scripts](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text#fine-tune)

+ CodeT5

  Following CodeT5, we fine-tune CodeT5 on our code summarization dataset using their [provided scripts](https://github.com/salesforce/CodeT5#fine-tuning).

### Code Generation 

+ CodeGPT

  Following CodeXGLEU, we fine-tune CodeGPT on our code generation dataset using their [provided scripts](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code#fine-tune)

+ CodeT5

  Following CodeT5, we fine-tune CodeT5 on our code generation dataset using their [provided scripts](https://github.com/salesforce/CodeT5#fine-tuning).

## Parameter-efficient Tuning

Following the documentation of [adapter-transformers](https://docs.adapterhub.ml/methods.html), we insert adapters to the studied pre-trained CLMs (CodeT5) and train the parameters of the adapters.

+ Prefix-tuning: https://docs.adapterhub.ml/methods.html#prefix-tuning
+ Union: https://docs.adapterhub.ml/method_combinations.html#method-combinations

## Zero/few-shot learning

The template NL-Code pairs are available at ```./zero&few-shot-test/template_prompt```

+ InCoder: Our implementation is based on the [scripts](https://github.com/dpfried/incoder#usage) provided by their GitHub repository.

+ CodeGen: Our implementation is based on the [scripts](https://github.com/salesforce/CodeGen#sampling-with-huggingface) provided by their GitHub repository.

+ StarCoder: Our implementation is based on the API provided by [Hugging Face](https://huggingface.co/bigcode/starcoder).

+ Code Llama: Our implementation is based on the API provided by [Hugging Face](https://huggingface.co/codellama/CodeLlama-13b-hf).

+ OpenAI GPT-3.5 & GPT-4: Our implementation is based on the [example scripts](https://platform.openai.com/docs/api-reference/completions/create) offered by OpenAI.

The implementation scripts of the above models are available at: ```./zero&few-shot-test/``` 