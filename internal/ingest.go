package internal

import (

	"os"
)

func chunkText(text string, size int) []string {
	var chunks []string 
	return chunks
}
// IngestFile now assumes GetEmbedding/VectorDB are passed in or in a shared package
func IngestFile(filePath string, getEmb func(string) []float32, upsert func([]float32, string)){
	//Read file
content, err := os.ReadFile(filePath)
	if err != nil {
		return
	}
	  //2 . break the content into chunks (eg.., 500 characters)
  chunks := chunkText(string(content),500)
  for _, chunks := range chunks{
	// 3. Get Embedding from OpenAI
	  vector := getEmb(chunks)
	  //4. save to vector DB
	  upsert(vector,chunks)
  }
}