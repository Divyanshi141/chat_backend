package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/dslipak/pdf"
	"github.com/gorilla/websocket"
	"github.com/sashabaranov/go-openai"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// DocumentChunk stores text locally for retrieval
type DocumentChunk struct {
	Text string
}

type VectorStore struct {
	chunks []DocumentChunk
}

var VectorDB = &VectorStore{}

// Upsert adds text chunks to memory locally
func (vs *VectorStore) Upsert(text string) {
	vs.chunks = append(vs.chunks, DocumentChunk{Text: text})
}

// Query performs a keyword-overlap search locally
func (vs *VectorStore) Query(query string, k int) []string {
	if len(vs.chunks) == 0 {
		return []string{}
	}

	queryWords := strings.Fields(strings.ToLower(query))

	type ScoredDoc struct {
		Text  string
		Score int
	}

	var scored []ScoredDoc
	for _, chunk := range vs.chunks {
		lowerText := strings.ToLower(chunk.Text)
		score := 0
		for _, word := range queryWords {
			if len(word) > 2 && strings.Contains(lowerText, word) {
				score++
			}
		}
		if score > 0 {
			scored = append(scored, ScoredDoc{Text: chunk.Text, Score: score})
		}
	}

	// Sort by highest match score
	for i := 0; i < len(scored)-1; i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[i].Score < scored[j].Score {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}

	var results []string
	limit := k
	if len(scored) < k {
		limit = len(scored)
	}

	for i := 0; i < limit; i++ {
		results = append(results, scored[i].Text)
	}
	return results
}

func handleUpload(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	file, _, err := r.FormFile("pdf_file")
	if err != nil {
		http.Error(w, "Failed to get file from request", http.StatusBadRequest)
		return
	}
	defer file.Close()

	pdfBytes, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "Failed to read file bytes", http.StatusInternalServerError)
		return
	}

	reader, err := pdf.NewReader(bytes.NewReader(pdfBytes), int64(len(pdfBytes)))
	if err != nil {
		http.Error(w, "Could not initialize PDF reader", http.StatusInternalServerError)
		return
	}

	plainTextReader, err := reader.GetPlainText()
	if err != nil {
		http.Error(w, "Could not extract text from PDF", http.StatusInternalServerError)
		return
	}

	textBytes, err := io.ReadAll(plainTextReader)
	if err != nil {
		http.Error(w, "Could not read plain text buffer", http.StatusInternalServerError)
		return
	}

	text := string(textBytes)
	VectorDB.chunks = []DocumentChunk{} // Reset on new upload

	chunkSize := 300
	for i := 0; i < len(text); i += chunkSize {
		end := i + chunkSize
		if end > len(text) {
			end = len(text)
		}
		chunk := strings.TrimSpace(text[i:end])
		if len(chunk) > 0 {
			VectorDB.Upsert(chunk)
		}
	}

	log.Println("New PDF successfully uploaded and indexed!")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("PDF uploaded successfully"))
}

func retrieveRelevantDocs(query string) string {
	results := VectorDB.Query(query, 3)
	if len(results) == 0 {
		return ""
	}
	return strings.Join(results, "\n---\n")
}

// streamGroqResponse calls Groq and streams chunks directly into the WebSocket
//stream groq add full conversational history
func streamGroqResponse(ws *websocket.Conn, history *[]openai.ChatCompletionMessage) {
	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		sendJSON(ws, map[string]interface{}{
			"status":  http.StatusUnauthorized,
			"type":    "error",
			"message": "GROQ_API_KEY is not configured in .env",
		})
		return
	}

	config := openai.DefaultConfig(apiKey)
	config.BaseURL = "https://api.groq.com/openai/v1"
	client := openai.NewClientWithConfig(config)

	model :=os.Getenv("GROQ_MODEL")
	if model == "" {
		model = "qwen/qwen3.8-27b"
	}
	// 1. Tell Flutter generation has started
	sendJSON(ws, map[string]interface{}{
		"status": 200,
		"type":   "start",
	})

		// 2. Request stream from Groq passing full conversation history
	stream, err := client.CreateChatCompletionStream(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: model,
			Messages: *history,
			Stream: true,
		},
	)
	if err != nil {
		log.Println("Groq Stream Error:", err)
		sendJSON(ws, map[string]interface{}{
			"status":  http.StatusInternalServerError,
			"type":    "error",
			"message": fmt.Sprintf("Groq API error: %v", err),
		})
		return
	}
	defer stream.Close()

    var fullResponse strings.Builder
	// 3. Forward each token to Flutter in real-time
	for {
		response, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break // Finished streaming
		}
		if err != nil {
			log.Println("Error receiving chunk:", err)
			break
		}

		delta := response.Choices[0].Delta.Content
		if delta != "" {
			fullResponse.WriteString(delta)
			sendJSON(ws, map[string]interface{}{
				"status":  200,
				"type":    "chunk",
				"message": delta,
			})
		}
	}
//Save AI's response to history for next turn's memory 
if fullResponse.Len() > 0 {
		*history = append(*history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: fullResponse.String(),
		})
	}

	// 5. Keep last 15 messages so context doesn't exceed token limits
	if len(*history) > 15 {
		// Keep system message at index 0, and retain last 14 messages
		trimmed := append([]openai.ChatCompletionMessage{(*history)[0]}, (*history)[len(*history)-14:]...)
		*history = trimmed
	}
	// 6. Signal completion to Flutter
	sendJSON(ws, map[string]interface{}{
		"status": 200,
		"type":   "done",
	})
}

func sendJSON(ws *websocket.Conn, payload map[string]interface{}) {
	data, err := json.Marshal(payload)
	if err != nil {
		return
	}
	_ = ws.WriteMessage(websocket.TextMessage, data)
}
func handleConnections(w http.ResponseWriter, r *http.Request) {
	ws, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Print("Upgrade error:", err)
		return
	}
	defer ws.Close()

	// 1. Initialize history with a System Prompt for this session
	history := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are a fast, knowledgeable AI assistant. Answer clearly and concisely.",
		},
	}

	for {
		_, p, err := ws.ReadMessage()
		if err != nil {
			log.Println("WebSocket read error:", err)
			break
		}

		rawMessage := strings.TrimSpace(string(p))
		if rawMessage == "" {
			continue
		}

		// Support "clear" command to reset memory when user clears chat
		if rawMessage == `{"type":"clear"}` || rawMessage == "clear" {
			history = []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleSystem,
					Content: "You are a fast, knowledgeable AI assistant. Answer clearly and concisely.",
				},
			}
			log.Println("Conversation memory reset")
			continue
		}

		fmt.Printf("User: %s\n", rawMessage)

		// 2. Add user query to history (OUTSIDE the clear block)
		history = append(history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: rawMessage,
		})

		// 3. Stream Groq response with full context memory
		streamGroqResponse(ws, &history)
	}
}

func loadEnv(filepath string) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return
	}
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			val := strings.TrimSpace(parts[1])
			val = strings.Trim(val, `"'`)
			if os.Getenv(key) == "" {
				os.Setenv(key, val)
			}
		}
	}
}

func main() {
	loadEnv(".env")
	http.HandleFunc("/upload", handleUpload)
	http.HandleFunc("/ws", handleConnections)

	log.Println("Server started on :8080")
	if err := http.ListenAndServe("0.0.0.0:8080", nil); err != nil {
		log.Fatal("ListenAndServe: ", err)
	}
}
