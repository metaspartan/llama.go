package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"
	"unicode"
)

// ----------------------------------------------------------------------------
// Data structures for Llama-3

type Config struct {
	Dim       int32 // transformer dimension
	HiddenDim int32 // for ffn layers
	NLayers   int32 // number of layers
	NHeads    int32 // number of query heads
	NKVHeads  int32 // number of key/value heads (multiquery)
	VocabSize int32 // typically 4096
	SeqLen    int32 // max sequence length
}

type TransformerWeights struct {
	TokenEmbeddingTable []float32 // shape (vocab_size, dim)

	RmsAttWeight []float32 // shape (layer, dim)
	RmsFfnWeight []float32 // shape (layer, dim)

	Wq []float32 // shape (layer, dim, n_heads*head_size)
	Wk []float32 // shape (layer, dim, n_kv_heads*head_size)
	Wv []float32 // shape (layer, dim, n_kv_heads*head_size)
	Wo []float32 // shape (layer, n_heads*head_size, dim)

	W1 []float32 // shape (layer, hidden_dim, dim)
	W2 []float32 // shape (layer, dim, hidden_dim)
	W3 []float32 // shape (layer, hidden_dim, dim)

	RmsFinalWeight []float32 // shape (dim,)

	// optional classifier
	Wcls []float32
}

type RunState struct {
	X      []float32
	Xb     []float32
	Xb2    []float32
	Hb     []float32
	Hb2    []float32
	Q      []float32
	Att    []float32
	Logits []float32

	KeyCache   []float32
	ValueCache []float32
}

type Transformer struct {
	Config   Config
	Weights  TransformerWeights
	State    RunState
	FullData []byte // entire model file
}

// ----------------------------------------------------------------------------
// Allocation

func mallocRunState(s *RunState, cfg Config) {
	dim := int(cfg.Dim)
	hiddenDim := int(cfg.HiddenDim)
	nHeads := int(cfg.NHeads)
	nKVHeads := int(cfg.NKVHeads)
	seqLen := int(cfg.SeqLen)
	kvDim := (dim * nKVHeads) / nHeads

	s.X = make([]float32, dim)
	s.Xb = make([]float32, dim)
	s.Xb2 = make([]float32, dim)
	s.Hb = make([]float32, hiddenDim)
	s.Hb2 = make([]float32, hiddenDim)
	s.Q = make([]float32, dim)
	s.Att = make([]float32, nHeads*seqLen)
	s.Logits = make([]float32, cfg.VocabSize)

	s.KeyCache = make([]float32, int(cfg.NLayers)*seqLen*kvDim)
	s.ValueCache = make([]float32, int(cfg.NLayers)*seqLen*kvDim)
}

// ----------------------------------------------------------------------------
// Reading the checkpoint

func readCheckpoint(
	checkpointPath string,
	cfg *Config,
	w *TransformerWeights,
) ([]byte, error) {

	data, err := os.ReadFile(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("cannot open %s: %v", checkpointPath, err)
	}
	if len(data) < 7*4 {
		return nil, errors.New("file too small to hold Config")
	}
	buf := bytes.NewReader(data[:7*4])
	if err := binary.Read(buf, binary.LittleEndian, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %v", err)
	}
	shared := true
	if cfg.VocabSize < 0 {
		shared = false
		cfg.VocabSize = -cfg.VocabSize
	}
	floatData := data[7*4:]

	// helper to read slices
	offset := 0
	floatSlice := func(count int) ([]float32, error) {
		bytesNeeded := count * 4
		if offset+bytesNeeded > len(floatData) {
			return nil, errors.New("not enough data left in file")
		}
		out := make([]float32, count)
		if err := binary.Read(
			bytes.NewReader(floatData[offset:offset+bytesNeeded]),
			binary.LittleEndian,
			out,
		); err != nil {
			return nil, err
		}
		offset += bytesNeeded
		return out, nil
	}

	dim := int(cfg.Dim)
	nLayers := int(cfg.NLayers)
	vocabSize := int(cfg.VocabSize)
	nHeads := int(cfg.NHeads)
	nKVHeads := int(cfg.NKVHeads)
	headSize := dim / nHeads

	var slice []float32
	// same order as memory_map_weights
	slice, err = func() ([]float32, error) { return floatSlice(vocabSize * dim) }()
	if err != nil {
		return nil, err
	}
	w.TokenEmbeddingTable = slice

	slice, err = floatSlice(nLayers * dim)
	if err != nil {
		return nil, err
	}
	w.RmsAttWeight = slice

	slice, err = floatSlice(nLayers * dim * (nHeads * headSize))
	if err != nil {
		return nil, err
	}
	w.Wq = slice

	slice, err = floatSlice(nLayers * dim * (nKVHeads * headSize))
	if err != nil {
		return nil, err
	}
	w.Wk = slice

	slice, err = floatSlice(nLayers * dim * (nKVHeads * headSize))
	if err != nil {
		return nil, err
	}
	w.Wv = slice

	slice, err = floatSlice(nLayers * (nHeads * headSize) * dim)
	if err != nil {
		return nil, err
	}
	w.Wo = slice

	slice, err = floatSlice(nLayers * dim)
	if err != nil {
		return nil, err
	}
	w.RmsFfnWeight = slice

	hiddenDim := int(cfg.HiddenDim)
	slice, err = floatSlice(nLayers * dim * hiddenDim)
	if err != nil {
		return nil, err
	}
	w.W1 = slice

	slice, err = floatSlice(nLayers * hiddenDim * dim)
	if err != nil {
		return nil, err
	}
	w.W2 = slice

	slice, err = floatSlice(nLayers * dim * hiddenDim)
	if err != nil {
		return nil, err
	}
	w.W3 = slice

	slice, err = floatSlice(dim)
	if err != nil {
		return nil, err
	}
	w.RmsFinalWeight = slice

	// skip freq_cis for rope:
	skipCount := (int(cfg.SeqLen) * headSize / 2) * 2
	offset += skipCount * 4
	if offset > len(floatData) {
		return nil, errors.New("not enough data to skip rope freq arrays")
	}

	// Wcls
	if shared {
		w.Wcls = w.TokenEmbeddingTable
	} else {
		slice, err = floatSlice(vocabSize * dim)
		if err != nil {
			return nil, err
		}
		w.Wcls = slice
	}

	return data, nil
}

func buildTransformer(t *Transformer, checkpointPath string) error {
	var err error
	t.FullData, err = readCheckpoint(checkpointPath, &t.Config, &t.Weights)
	if err != nil {
		return err
	}
	mallocRunState(&t.State, t.Config)
	return nil
}

// ----------------------------------------------------------------------------
// Math / forward pass

func rmsnorm(o, x, weight []float32, size int) {
	var ss float32
	for i := 0; i < size; i++ {
		ss += x[i] * x[i]
	}
	ss /= float32(size)
	ss += 1.0e-5
	inv := float32(1.0 / math.Sqrt(float64(ss)))

	for i := 0; i < size; i++ {
		o[i] = weight[i] * (inv * x[i])
	}
}

func softmax(x []float32, size int) {
	maxVal := x[0]
	for i := 1; i < size; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	var sum float32
	for i := 0; i < size; i++ {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += x[i]
	}
	for i := 0; i < size; i++ {
		x[i] /= sum
	}
}

func matmul(xout, x, w []float32, n, d int) {
	for i := 0; i < d; i++ {
		val := float32(0.0)
		rowOffset := i * n
		for j := 0; j < n; j++ {
			val += w[rowOffset+j] * x[j]
		}
		xout[i] = val
	}
}

func forward(t *Transformer, token, pos int) []float32 {
	p := t.Config
	w := t.Weights
	s := &t.State

	dim := int(p.Dim)
	nHeads := int(p.NHeads)
	nKVHeads := int(p.NKVHeads)
	hiddenDim := int(p.HiddenDim)
	headSize := dim / nHeads
	kvDim := (dim * nKVHeads) / nHeads
	kvMul := nHeads / nKVHeads

	x := s.X
	copy(x, w.TokenEmbeddingTable[token*dim:token*dim+dim])

	// layers
	for layer := 0; layer < int(p.NLayers); layer++ {
		// RMSNorm (att)
		rmsnorm(s.Xb, x, w.RmsAttWeight[layer*dim:layer*dim+dim], dim)

		loff := layer * int(p.SeqLen) * kvDim
		kPtr := s.KeyCache[loff+pos*kvDim : loff+pos*kvDim+kvDim]
		vPtr := s.ValueCache[loff+pos*kvDim : loff+pos*kvDim+kvDim]

		// Q, K, V
		// w.Wq shape is [layer, dim, dim]
		matmul(s.Q, s.Xb, w.Wq[layer*dim*dim:layer*dim*dim+dim*dim], dim, dim)
		matmul(kPtr, s.Xb, w.Wk[layer*dim*kvDim:layer*dim*kvDim+dim*kvDim], dim, kvDim)
		matmul(vPtr, s.Xb, w.Wv[layer*dim*kvDim:layer*dim*kvDim+dim*kvDim], dim, kvDim)

		// rope
		for i := 0; i < nHeads; i++ {
			for j := 0; j < headSize; j += 2 {
				freq := float32(1.0 / math.Pow(500000.0, float64(j)/float64(headSize)))
				val := float32(pos) * freq
				c := float32(math.Cos(float64(val)))
				si := float32(math.Sin(float64(val)))
				// Q
				idxQ := i*headSize + j
				q0 := s.Q[idxQ]
				q1 := s.Q[idxQ+1]
				s.Q[idxQ] = q0*c - q1*si
				s.Q[idxQ+1] = q0*si + q1*c

				// K if within nKVHeads
				if i < nKVHeads {
					idxK := i*headSize + j
					k0 := kPtr[idxK]
					k1 := kPtr[idxK+1]
					kPtr[idxK] = k0*c - k1*si
					kPtr[idxK+1] = k0*si + k1*c
				}
			}
		}

		// multi-head attention
		// zero s.Xb
		for i := 0; i < dim; i++ {
			s.Xb[i] = 0
		}
		attPtr := s.Att
		for h := 0; h < nHeads; h++ {
			qh := s.Q[h*headSize : h*headSize+headSize]
			attH := attPtr[h*int(p.SeqLen) : h*int(p.SeqLen)+int(p.SeqLen)]
			for tpos := 0; tpos <= pos; tpos++ {
				kOff := loff + tpos*kvDim + (h/kvMul)*headSize
				kk := s.KeyCache[kOff : kOff+headSize]
				// dot
				var sc float32
				for i := 0; i < headSize; i++ {
					sc += qh[i] * kk[i]
				}
				sc /= float32(math.Sqrt(float64(headSize)))
				attH[tpos] = sc
			}
			softmax(attH, pos+1)

			xbH := s.Xb[h*headSize : h*headSize+headSize]
			for tpos := 0; tpos <= pos; tpos++ {
				vOff := loff + tpos*kvDim + (h/kvMul)*headSize
				vv := s.ValueCache[vOff : vOff+headSize]
				alpha := attH[tpos]
				for i := 0; i < headSize; i++ {
					xbH[i] += alpha * vv[i]
				}
			}
		}

		// output matmul
		matmul(s.Xb2, s.Xb, w.Wo[layer*dim*dim:layer*dim*dim+dim*dim], dim, dim)
		for i := 0; i < dim; i++ {
			x[i] += s.Xb2[i]
		}

		// RMSNorm (ffn)
		rmsnorm(s.Xb, x, w.RmsFfnWeight[layer*dim:layer*dim+dim], dim)

		// w1, w3 => s.Hb, s.Hb2
		matmul(s.Hb, s.Xb, w.W1[layer*dim*hiddenDim:layer*dim*hiddenDim+dim*hiddenDim], dim, hiddenDim)
		matmul(s.Hb2, s.Xb, w.W3[layer*dim*hiddenDim:layer*dim*hiddenDim+dim*hiddenDim], dim, hiddenDim)

		// SwiGLU
		for i := 0; i < hiddenDim; i++ {
			val := s.Hb[i]
			// silu
			val = val * (1.0 / (1.0 + float32(math.Exp(float64(-val)))))
			val *= s.Hb2[i]
			s.Hb[i] = val
		}

		matmul(s.Xb, s.Hb, w.W2[layer*dim*hiddenDim:layer*dim*hiddenDim+hiddenDim*dim], hiddenDim, dim)
		for i := 0; i < dim; i++ {
			x[i] += s.Xb[i]
		}
	}

	// final RMSNorm
	rmsnorm(x, x, w.RmsFinalWeight, dim)

	// classifier
	matmul(s.Logits, x, w.Wcls, dim, int(p.VocabSize))
	return s.Logits
}

// ----------------------------------------------------------------------------
// Tokenizer: now with triple merges, new BOS=128000, EOS=128001, etc.

type TokenIndex struct {
	Str string
	Id  int
}

type Tokenizer struct {
	Vocab       []string
	VocabScores []float32
	Sorted      []TokenIndex
	VocabSize   int
	MaxTokenLen int
	BytePieces  [512]byte
}

// buildTokenizer loads the file with the new format
func buildTokenizer(t *Tokenizer, path string, vocabSize int32) error {
	// Regular tokens (0 to vocabSize-1) plus special tokens (128000-128255)
	t.Vocab = make([]string, 128256) // Make room for all possible tokens
	t.VocabScores = make([]float32, 128256)
	t.VocabSize = int(vocabSize)

	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("cannot open tokenizer: %v", err)
	}
	defer f.Close()

	// fill BytePieces
	for i := 0; i < 256; i++ {
		t.BytePieces[i*2] = byte(i)
		t.BytePieces[i*2+1] = 0
	}

	// Read max_token_len
	var maxTokenLen int32
	if err := binary.Read(f, binary.LittleEndian, &maxTokenLen); err != nil {
		return fmt.Errorf("failed read max_token_length: %v", err)
	}
	t.MaxTokenLen = int(maxTokenLen)

	// Read regular tokens (0 to vocabSize)
	for i := 0; i < int(vocabSize); i++ {
		if err := binary.Read(f, binary.LittleEndian, &t.VocabScores[i]); err != nil {
			return fmt.Errorf("failed read vocab score: %v", err)
		}
		var length int32
		if err := binary.Read(f, binary.LittleEndian, &length); err != nil {
			return fmt.Errorf("failed read length: %v", err)
		}
		if length < 0 || length > 100000 {
			return fmt.Errorf("invalid token length: %d", length)
		}
		buf := make([]byte, length)
		if _, err := io.ReadFull(f, buf); err != nil {
			return fmt.Errorf("failed read token: %v", err)
		}
		t.Vocab[i] = string(buf)
	}

	// Add Llama 3.2 special tokens
	specialTokens := map[int]string{
		128000: "<|im_start|>",
		128001: "<|im_end|>",
		128002: "<s>",
		128003: "</s>",
	}

	for id, token := range specialTokens {
		t.Vocab[id] = token
		t.VocabScores[id] = 0.0
	}

	return nil
}

func (t *Tokenizer) ensureSorted() {
	if t.Sorted != nil {
		return
	}

	// Count total tokens (regular + special)
	count := 0
	for i := range t.Vocab {
		if t.Vocab[i] != "" {
			count++
		}
	}

	// Create sorted array
	t.Sorted = make([]TokenIndex, 0, count)

	// Add all non-empty tokens (both regular and special)
	for i := range t.Vocab {
		if t.Vocab[i] != "" {
			t.Sorted = append(t.Sorted, TokenIndex{Str: t.Vocab[i], Id: i})
		}
	}

	// Sort by string content
	sort.Slice(t.Sorted, func(i, j int) bool {
		return t.Sorted[i].Str < t.Sorted[j].Str
	})
}

func (t *Tokenizer) strLookup(s string) int {
	t.ensureSorted()
	lo, hi := 0, len(t.Sorted) // Changed from t.VocabSize to len(t.Sorted)
	for lo < hi {
		mid := (lo + hi) >> 1
		if t.Sorted[mid].Str < s {
			lo = mid + 1
		} else if t.Sorted[mid].Str > s {
			hi = mid
		} else {
			return t.Sorted[mid].Id
		}
	}
	return -1
}

func (t *Tokenizer) decode(_prevToken, token int) string {
	piece := t.Vocab[token]
	// raw bytes like <0x01>
	var bVal byte
	if len(piece) == 6 && strings.HasPrefix(piece, "<0x") && piece[5] == '>' {
		_, err := fmt.Sscanf(piece, "<0x%02X>", &bVal)
		if err == nil {
			// single byte
			return string(t.BytePieces[bVal*2 : bVal*2+1])
		}
	}
	return piece
}

func safePrintf(piece string) {
	if len(piece) == 1 {
		b := piece[0]
		if b == '\n' {
			fmt.Println()
			return
		}
		if !unicode.IsPrint(rune(b)) && !unicode.IsSpace(rune(b)) {
			return
		}
	}
	fmt.Print(piece)
}

// The triple-merge logic from llama3.c means after we parse codepoints or bytes,
// we attempt to find either pair merges or triple merges with best score.
func (t *Tokenizer) encode(text string, bos, eos bool, tokens []int) (int, error) {
	n := 0

	// Add BOS token if requested (128002 is <s> in Llama 3.2)
	if bos {
		tokens[n] = 128002
		n++
	}

	// Add space prefix if text not empty
	if text != "" {
		spID := t.strLookup(" ")
		if spID < 0 {
			return n, fmt.Errorf("couldn't find space token in vocab")
		}
		tokens[n] = spID
		n++
	}

	// Process text runes
	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		codepoint := string(runes[i])
		idx := t.strLookup(codepoint)

		if idx >= 0 {
			tokens[n] = idx
			n++
		} else {
			// Fallback to bytes
			b := []byte(string(runes[i]))
			for _, bb := range b {
				tokens[n] = int(bb) + 3
				n++
			}
		}
	}

	// Add EOS token if requested (128003 is </s> in Llama 3.2)
	if eos {
		tokens[n] = 128003
		n++
	}

	if n == 0 {
		return 0, fmt.Errorf("no tokens generated from text: %q", text)
	}

	return n, nil
}

// ----------------------------------------------------------------------------
// Sampler

type ProbIndex struct {
	Prob  float32
	Index int
}

type Sampler struct {
	VocabSize   int
	ProbIndex   []ProbIndex
	Temperature float32
	TopP        float32
	Rng         *rand.Rand
}

func buildSampler(vocabSize int, temperature, topp float32, seed int64) Sampler {
	rng := rand.New(rand.NewSource(seed))
	return Sampler{
		VocabSize:   vocabSize,
		ProbIndex:   make([]ProbIndex, vocabSize),
		Temperature: temperature,
		TopP:        topp,
		Rng:         rng,
	}
}

func sampleArgmax(probs []float32, n int) int {
	bestI := 0
	bestP := probs[0]
	for i := 1; i < n; i++ {
		if probs[i] > bestP {
			bestP = probs[i]
			bestI = i
		}
	}
	return bestI
}

func sampleMult(probs []float32, n int, coin float32) int {
	var cdf float32
	for i := 0; i < n; i++ {
		cdf += probs[i]
		if coin < cdf {
			return i
		}
	}
	return n - 1
}

func (s *Sampler) sampleTopP(probs []float32, n int, coin float32) int {
	cutoff := (1.0 - s.TopP) / float32(n-1)
	n0 := 0
	for i := 0; i < n; i++ {
		if probs[i] >= cutoff {
			s.ProbIndex[n0].Index = i
			s.ProbIndex[n0].Prob = probs[i]
			n0++
		}
	}
	sort.Slice(s.ProbIndex[:n0], func(i, j int) bool {
		return s.ProbIndex[i].Prob > s.ProbIndex[j].Prob
	})
	var cum float32
	last := n0 - 1
	for i := 0; i < n0; i++ {
		cum += s.ProbIndex[i].Prob
		if cum > s.TopP {
			last = i
			break
		}
	}
	r := coin * cum
	var cdf float32
	for i := 0; i <= last; i++ {
		cdf += s.ProbIndex[i].Prob
		if r < cdf {
			return s.ProbIndex[i].Index
		}
	}
	return s.ProbIndex[last].Index
}

func (s *Sampler) sample(logits []float32) int {
	if s.Temperature == 0.0 {
		return sampleArgmax(logits, s.VocabSize)
	}
	// apply temperature
	for i := 0; i < s.VocabSize; i++ {
		logits[i] /= s.Temperature
	}
	softmax(logits, s.VocabSize)
	coin := s.Rng.Float32()
	if s.TopP <= 0.0 || s.TopP >= 1.0 {
		return sampleMult(logits, s.VocabSize, coin)
	}
	return s.sampleTopP(logits, s.VocabSize, coin)
}

// ----------------------------------------------------------------------------
// utility

func timeMs() int64 {
	return time.Now().UnixNano() / 1_000_000
}

// ----------------------------------------------------------------------------
// generate

func generate(transformer *Transformer, tokenizer *Tokenizer, sampler *Sampler, prompt string, steps int) {
	if prompt == "" {
		prompt = ""
	}
	tokens := make([]int, len(prompt)+512)

	fmt.Printf("Encoding prompt: %s\n", prompt)
	fmt.Printf("Tokens: %v\n", tokens)
	n, err := tokenizer.encode(prompt, true, false, tokens)
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode error: %v\n", err)
		return
	}
	if n < 1 {
		fmt.Fprintf(os.Stderr, "no tokens?\n")
		return
	}
	var start int64
	token := tokens[0]
	pos := 0
	for pos < steps {
		logits := forward(transformer, token, pos)
		var next int
		if pos < (n - 1) {
			next = tokens[pos+1]
		} else {
			next = sampler.sample(logits)
		}
		pos++
		// if next is 128001 or 128009 => EOS, break
		if (next == 128001 || next == 128009) && pos >= n {
			break
		}
		piece := tokenizer.decode(token, next)
		safePrintf(piece)
		token = next
		if start == 0 {
			start = timeMs()
		}
	}
	fmt.Println()
	if pos > 1 {
		end := timeMs()
		speed := float64(pos-1) / (float64(end-start) / 1000.0)
		fmt.Fprintf(os.Stderr, "achieved tok/s: %f\n", speed)
	}
}

// ----------------------------------------------------------------------------
// chat

func readLine(guide string) string {
	fmt.Print(guide)
	scanner := bufio.NewScanner(os.Stdin)
	if !scanner.Scan() {
		return ""
	}
	return scanner.Text()
}

func chat(transformer *Transformer, tokenizer *Tokenizer, sampler *Sampler,
	cliUserPrompt, cliSystemPrompt string, steps int) {

	// We replicate the big chunk from llama3.c (the special tokens etc.).
	// For brevity, we do a simpler approach: we just encode user prompt => assistant output => repeated.

	sysPrompt := ""
	userPrompt := ""

	promptTokens := make([]int, 65536)
	systemPromptTokens := make([]int, 65536)
	userPromptTokens := make([]int, 65536)

	pos := 0
	userTurn := true
	var token, next int
	var nPrompt int

	for pos < steps {
		if userTurn {
			if pos == 0 {
				if cliSystemPrompt == "" {
					sysPrompt = readLine("Enter system prompt (optional): ")
				} else {
					sysPrompt = cliSystemPrompt
				}
				// We can put fancy tokens like <|begin_of_text|> = 128000, etc.
				// For simplicity: encode system prompt as plain text
				nSys, _ := tokenizer.encode(sysPrompt, false, false, systemPromptTokens)
				copy(promptTokens, systemPromptTokens[:nSys])
				nPrompt = nSys
			} else {
				nPrompt = 0 // re-start each turn
			}
			if pos == 0 && cliUserPrompt != "" {
				userPrompt = cliUserPrompt
			} else {
				userPrompt = readLine("User (or exit): ")
				if userPrompt == "exit" {
					break
				}
			}
			nUser, _ := tokenizer.encode(userPrompt, false, false, userPromptTokens)
			copy(promptTokens[nPrompt:], userPromptTokens[:nUser])
			nPrompt += nUser
			userTurn = false
			fmt.Print("Assistant: ")
		}

		// pick next token from prompt or sampling
		if nPrompt > 0 {
			token = promptTokens[0]
			copy(promptTokens, promptTokens[1:nPrompt])
			nPrompt--
		} else {
			token = next
		}

		logits := forward(transformer, token, pos)
		next = sampler.sample(logits)
		pos++
		// if next is an EOS token, userTurn = true
		if next == 128001 || next == 128009 {
			fmt.Println()
			userTurn = true
			continue
		}
		piece := tokenizer.decode(token, next)
		safePrintf(piece)
	}
	fmt.Println()
}

// ----------------------------------------------------------------------------
// CLI

func usage() {
	exe := os.Args[0]
	fmt.Fprintf(os.Stderr, "Usage: %s <checkpoint> [options]\n", exe)
	fmt.Fprintf(os.Stderr, "Example: %s model.bin -n 4096 -i \"Once upon a time\"\n", exe)
	fmt.Fprintf(os.Stderr, "Options:\n")
	fmt.Fprintf(os.Stderr, "  -t <float>   temperature, default 1.0\n")
	fmt.Fprintf(os.Stderr, "  -p <float>   top-p, default 0.9\n")
	fmt.Fprintf(os.Stderr, "  -s <int>     random seed, default = time.Now().Unix()\n")
	fmt.Fprintf(os.Stderr, "  -n <int>     steps, default 4096, 0 => config.seq_len\n")
	fmt.Fprintf(os.Stderr, "  -i <string>  input prompt\n")
	fmt.Fprintf(os.Stderr, "  -z <string>  tokenizer path, default tokenizer.bin\n")
	fmt.Fprintf(os.Stderr, "  -m <string>  mode: generate|chat, default generate\n")
	fmt.Fprintf(os.Stderr, "  -y <string>  system prompt in chat mode\n")
}

var (
	temperature   = flag.Float64("t", 1.0, "temperature")
	topP          = flag.Float64("p", 0.9, "top-p")
	seed          = flag.Int64("s", 0, "random seed")
	steps         = flag.Int("n", 4096, "steps")
	promptStr     = flag.String("i", "", "input prompt")
	tokenizerPath = flag.String("z", "tokenizer.bin", "tokenizer path")
	mode          = flag.String("m", "generate", "mode")
	systemPrompt  = flag.String("y", "", "system prompt for chat mode")
)

func main() {
	flag.Usage = usage
	flag.Parse()
	args := flag.Args()
	if len(args) < 1 {
		usage()
		os.Exit(1)
	}
	checkpointPath := args[0]

	if *seed <= 0 {
		*seed = time.Now().Unix()
	}
	if *temperature < 0 {
		*temperature = 0
	}
	if *topP < 0.0 || *topP > 1.0 {
		*topP = 0.9
	}

	var T Transformer
	if err := buildTransformer(&T, checkpointPath); err != nil {
		fmt.Fprintf(os.Stderr, "buildTransformer error: %v\n", err)
		os.Exit(1)
	}
	if *steps <= 0 || *steps > int(T.Config.SeqLen) {
		*steps = int(T.Config.SeqLen)
	}

	var tokenizer Tokenizer
	if err := buildTokenizer(&tokenizer, *tokenizerPath, T.Config.VocabSize); err != nil {
		fmt.Fprintf(os.Stderr, "buildTokenizer error: %v\n", err)
		os.Exit(1)
	}

	sampler := buildSampler(int(T.Config.VocabSize), float32(*temperature), float32(*topP), *seed)

	if *mode == "generate" {
		generate(&T, &tokenizer, &sampler, *promptStr, *steps)
	} else if *mode == "chat" {
		chat(&T, &tokenizer, &sampler, *promptStr, *systemPrompt, *steps)
	} else {
		fmt.Fprintf(os.Stderr, "unknown mode: %s\n", *mode)
		usage()
	}
}
