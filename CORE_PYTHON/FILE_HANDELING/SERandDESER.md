Serialization and Deserialization 
    
    `     serialization is the process of converting the complex data in python in JSON format 
        Deserialization is the reverse proecess of the Serialization 
        It's  the reverse  Process of the Deserialization`

**📊 Diagram — Serialization pipeline: object to bytes to file and back**

```mermaid
flowchart LR
    OBJ["Python object<br/>dict - list - custom class"] --> D["pickle.dumps(obj)"]
    D --> BY["bytes object"]
    BY --> W["write to file<br/>open mode wb"]
    W --> F["file on disk"]
    F --> R["read file<br/>open mode rb"]
    R --> L["pickle.loads(data)"]
    L --> OBJ2["Python object back<br/>same data, new identity"]
```

**📊 Diagram — JSON vs Pickle: which one to pick**

```mermaid
flowchart TD
    DATA["Python data to save or send"] --> Q{"Need human-readable text<br/>or cross-language support?"}
    Q -->|"Yes"| JS["JSON - json module"]
    JS --> JSL["json.dumps / json.load<br/>text format - readable anywhere<br/>only dict, list, str, number, bool, None"]
    Q -->|"No - Python only"| PK["Pickle - pickle module"]
    PK --> PKL["pickle.dumps / pickle.loads<br/>almost ANY Python object<br/>bytes - not human readable"]
    JSL --> OUT["Serialized output saved or sent"]
    PKL --> OUT
```