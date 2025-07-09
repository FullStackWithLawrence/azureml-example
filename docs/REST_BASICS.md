# REST API Basics

REST (Representational State Transfer) is a standard for building web APIs using HTTP. Below are the core concepts with practical `curl` examples, using the OpenAI API as a reference.

## HTTP Verbs

- **GET**: Retrieve data.

  ```bash
  curl https://api.openai.com/v1/models \
    -H "Authorization: Bearer $OPENAI_API_KEY"
  ```

- **POST**: Create new resources.

  ```bash
  curl https://api.openai.com/v1/chat/completions \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}'
  ```

- **PUT**: Replace a resource.
- **PATCH**: Update part of a resource.
- **DELETE**: Remove a resource.

## Headers

- **Authorization**: Used for authentication.

  ```bash
  -H "Authorization: Bearer $OPENAI_API_KEY"
  ```

- **Content-Type**: Specifies the body format.
  - `application/json` for JSON data
  - `application/x-www-form-urlencoded` for form data
  - `multipart/form-data` for file uploads

## Body

- **JSON**: Most APIs accept JSON.

  ```bash
  -d '{"key":"value"}'
  ```

- **Form Data**:

  ```bash
  -d "key1=value1&key2=value2" \
  -H "Content-Type: application/x-www-form-urlencoded"
  ```

- **Binary/File Upload**:

  ```bash
  curl -F "file=@myfile.png" https://api.example.com/upload
  ```

## Parameters

- **Query Parameters**: Appended to the URL.

  ```bash
  curl "https://api.example.com/data?utm_source=github&utm_medium=readme"
  ```

- **Common Examples**:
  - `utm_source`, `utm_medium`, `utm_campaign` for analytics
  - `limit`, `offset` for pagination

---

**Pro Tip:** Always check the API documentation for required headers, body formats, and supported parameters.
