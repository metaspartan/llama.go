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
// Data structures

type Config struct {
	Dim       int32 // transformer dimension
	HiddenDim int32 // for ffn layers
	NLayers   int32 // number of layers
	NHeads    int32 // number of query heads
	NKVHeads  int32 // number of key/value heads (can be < query heads because of multiquery)
	VocabSize int32 // vocabulary size, usually 256 (byte-level)
	SeqLen    int32 // max sequence length
}

// TransformerWeights holds all the model parameters (embeddings, layer weights, etc.)
type TransformerWeights struct {
	// token embedding table
	TokenEmbeddingTable []float32 // shape (vocab_size, dim)

	// weights for rmsnorms
	RmsAttWeight []float32 // shape (layer, dim)
	RmsFfnWeight []float32 // shape (layer, dim)

	// weights for matmuls
	Wq []float32 // shape (layer, dim, n_heads * head_size)
	Wk []float32 // shape (layer, dim, n_kv_heads * head_size)
	Wv []float32 // shape (layer, dim, n_kv_heads * head_size)
	Wo []float32 // shape (layer, n_heads * head_size, dim)

	// ffn
	W1 []float32 // shape (layer, hidden_dim, dim)
	W2 []float32 // shape (layer, dim, hidden_dim)
	W3 []float32 // shape (layer, hidden_dim, dim)

	// final rmsnorm
	RmsFinalWeight []float32 // shape (dim,)

	// optional classifier
	Wcls []float32
}

// RunState holds the activations and any caches (K/V) for incremental inference
type RunState struct {
	X      []float32 // activation for current time stamp (dim,)
	Xb     []float32 // buffer same size as X
	Xb2    []float32 // additional buffer (dim,)
	Hb     []float32 // buffer for hidden dimension in the ffn (hidden_dim,)
	Hb2    []float32 // buffer for hidden dimension in the ffn (hidden_dim,)
	Q      []float32 // query (dim,)
	Att    []float32 // attention buffer (n_heads, seq_len)
	Logits []float32 // output logits

	KeyCache   []float32 // shape (layer, seq_len, kv_dim)
	ValueCache []float32 // shape (layer, seq_len, kv_dim)
}

// Transformer groups the Config, Weights, and the RunState
type Transformer struct {
	Config   Config
	Weights  TransformerWeights
	State    RunState
	FullData []byte // the entire model file read into a slice
}

// ----------------------------------------------------------------------------
// Allocation and freeing

func mallocRunState(s *RunState, cfg Config) {
	dim := int(cfg.Dim)
	hiddenDim := int(cfg.HiddenDim)
	nLayers := int(cfg.NLayers)
	nHeads := int(cfg.NHeads)
	nKVHeads := int(cfg.NKVHeads)
	seqLen := int(cfg.SeqLen)

	kvDim := (dim * nKVHeads) / nHeads // how many floats per key/value
	s.X = make([]float32, dim)
	s.Xb = make([]float32, dim)
	s.Xb2 = make([]float32, dim)
	s.Hb = make([]float32, hiddenDim)
	s.Hb2 = make([]float32, hiddenDim)
	s.Q = make([]float32, dim)
	s.Att = make([]float32, nHeads*seqLen)
	s.Logits = make([]float32, cfg.VocabSize)

	// KV caches
	s.KeyCache = make([]float32, nLayers*seqLen*kvDim)
	s.ValueCache = make([]float32, nLayers*seqLen*kvDim)
}

// ----------------------------------------------------------------------------
// Reading the checkpoint (pure Go approach: read entire file into memory)

func readCheckpoint(
	checkpointPath string,
	cfg *Config,
	w *TransformerWeights,
) ([]byte, error) {

	data, err := os.ReadFile(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("couldn't open file %s: %v", checkpointPath, err)
	}

	// first bytes correspond to the Config struct
	// which in C is laid out as 7 int32 fields
	if len(data) < 7*4 {
		return nil, errors.New("checkpoint file too small to contain Config")
	}

	// read the config
	buf := bytes.NewReader(data[:7*4])
	if err := binary.Read(buf, binary.LittleEndian, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config from checkpoint: %v", err)
	}

	// Negative vocab_size means unshared weights in the original code.
	// We'll replicate that logic (though typically we only see shared).
	sharedWeights := true
	if cfg.VocabSize < 0 {
		sharedWeights = false
		cfg.VocabSize = -cfg.VocabSize
	}

	// now the rest of the file is float32 data
	floatData := data[7*4:]
	// The model parameters are stored as float32. We must figure out how many floats:
	// totalFloats = (len(floatData) / 4). We'll treat them as []float32.

	// We'll parse them in the same order as memory_map_weights in the C code
	// but we store them in slices in Go.

	// We define a helper to get a sub-slice as floats:
	floatSlice := func(offset, count int) ([]float32, int, error) {
		byteCount := count * 4
		if offset+byteCount > len(floatData) {
			return nil, offset, errors.New("not enough float data in checkpoint")
		}
		// interpret the sub-slice
		raw := floatData[offset : offset+byteCount]
		// convert bytes to float32
		out := make([]float32, count)
		if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, out); err != nil {
			return nil, offset, err
		}
		return out, offset + byteCount, nil
	}

	// gather counts
	dim := int(cfg.Dim)
	nLayers := int(cfg.NLayers)
	vocabSize := int(cfg.VocabSize)
	hiddenDim := int(cfg.HiddenDim)
	nHeads := int(cfg.NHeads)
	nKVHeads := int(cfg.NKVHeads)
	headSize := dim / nHeads
	// kvDim := (dim * nKVHeads) / nHeads

	// same logic as in C memory_map_weights
	offset := 0

	var err2 error
	w.TokenEmbeddingTable, offset, err2 = floatSlice(offset, vocabSize*dim)
	if err2 != nil {
		return nil, err2
	}
	w.RmsAttWeight, offset, err2 = floatSlice(offset, nLayers*dim)
	if err2 != nil {
		return nil, err2
	}
	w.Wq, offset, err2 = floatSlice(offset, nLayers*dim*(nHeads*headSize))
	if err2 != nil {
		return nil, err2
	}
	w.Wk, offset, err2 = floatSlice(offset, nLayers*dim*(nKVHeads*headSize))
	if err2 != nil {
		return nil, err2
	}
	w.Wv, offset, err2 = floatSlice(offset, nLayers*dim*(nKVHeads*headSize))
	if err2 != nil {
		return nil, err2
	}
	w.Wo, offset, err2 = floatSlice(offset, nLayers*(nHeads*headSize)*dim)
	if err2 != nil {
		return nil, err2
	}
	w.RmsFfnWeight, offset, err2 = floatSlice(offset, nLayers*dim)
	if err2 != nil {
		return nil, err2
	}
	w.W1, offset, err2 = floatSlice(offset, nLayers*dim*hiddenDim)
	if err2 != nil {
		return nil, err2
	}
	w.W2, offset, err2 = floatSlice(offset, nLayers*hiddenDim*dim)
	if err2 != nil {
		return nil, err2
	}
	w.W3, offset, err2 = floatSlice(offset, nLayers*dim*hiddenDim)
	if err2 != nil {
		return nil, err2
	}
	w.RmsFinalWeight, offset, err2 = floatSlice(offset, dim)
	if err2 != nil {
		return nil, err2
	}

	// skip freq_cis_real, freq_cis_imag
	skipCount := (int(cfg.SeqLen) * headSize / 2) * 2
	offset += skipCount * 4
	if offset > len(floatData) {
		return nil, errors.New("not enough data to skip freq_cis")
	}

	// wcls
	if sharedWeights {
		// re-use token embedding
		w.Wcls = w.TokenEmbeddingTable
	} else {
		w.Wcls, offset, err2 = floatSlice(offset, vocabSize*dim)
		if err2 != nil {
			return nil, err2
		}
	}

	return data, nil
}

func buildTransformer(t *Transformer, checkpointPath string) error {
	var err error
	t.FullData, err = readCheckpoint(checkpointPath, &t.Config, &t.Weights)
	if err != nil {
		return err
	}
	// allocate run state
	mallocRunState(&t.State, t.Config)
	return nil
}

// ----------------------------------------------------------------------------
// Math / inference routines

func rmsnorm(o, x, weight []float32, size int) {
	// sum of squares
	ss := float32(0.0)
	for i := 0; i < size; i++ {
		ss += x[i] * x[i]
	}
	ss /= float32(size)
	ss += 1.0e-5
	ss = 1.0 / float32(math.Sqrt(float64(ss)))
	// normalize
	for i := 0; i < size; i++ {
		o[i] = weight[i] * (ss * x[i])
	}
}

func softmax(x []float32, size int) {
	// find max
	maxVal := x[0]
	for i := 1; i < size; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	sum := float32(0.0)
	for i := 0; i < size; i++ {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += x[i]
	}
	for i := 0; i < size; i++ {
		x[i] /= sum
	}
}

func matmul(xout, x, w []float32, n, d int) {
	// W (d,n) @ x (n,) -> xout (d,)
	// in the C code there's an #pragma omp parallel for.
	// We'll do a straightforward loop in Go. If desired, add concurrency.
	for i := 0; i < d; i++ {
		val := float32(0.0)
		rowOffset := i * n
		for j := 0; j < n; j++ {
			val += w[rowOffset+j] * x[j]
		}
		xout[i] = val
	}
}

func forward(transformer *Transformer, token, pos int) []float32 {
	p := transformer.Config
	w := transformer.Weights
	s := &transformer.State

	dim := int(p.Dim)
	nHeads := int(p.NHeads)
	nKVHeads := int(p.NKVHeads)
	kvDim := (dim * nKVHeads) / nHeads
	hiddenDim := int(p.HiddenDim)
	headSize := dim / nHeads
	kvMul := nHeads / nKVHeads

	x := s.X
	// copy token embedding into x
	copy(x, w.TokenEmbeddingTable[token*dim:token*dim+dim])

	// forward through all layers
	for l := 0; l < int(p.NLayers); l++ {
		// attention rmsnorm
		rmsnorm(s.Xb, x, w.RmsAttWeight[l*dim:l*dim+dim], dim)

		loff := l * int(p.SeqLen) * kvDim
		// K/V for this pos
		kPtr := s.KeyCache[loff+pos*kvDim : loff+pos*kvDim+kvDim]
		vPtr := s.ValueCache[loff+pos*kvDim : loff+pos*kvDim+kvDim]

		// matmul for q, k, v
		matmul(s.Q, s.Xb, w.Wq[l*dim*dim:l*dim*dim+dim*dim], dim, dim)
		matmul(kPtr, s.Xb, w.Wk[l*dim*kvDim:l*dim*kvDim+dim*kvDim], dim, kvDim)
		matmul(vPtr, s.Xb, w.Wv[l*dim*kvDim:l*dim*kvDim+dim*kvDim], dim, kvDim)

		// rope
		// The original code rotates pairs of floats in Q, K.
		// We do a direct port, ignoring actual details that Llama 2 might do.
		// In the C code: for (int i = 0; i < dim; i+=2) ...
		for i := 0; i < dim; i += 2 {
			headDim := i % headSize
			freq := float32(1.0 / math.Pow(10000.0, float64(headDim)/float64(headSize)))
			val := float32(pos) * freq
			fcr := float32(math.Cos(float64(val)))
			fci := float32(math.Sin(float64(val)))
			// how many vectors to rotate? If i < kv_dim => rotate Q & K, else rotate Q only
			rotn := 1
			if i < kvDim {
				rotn = 2
			}
			for v := 0; v < rotn; v++ {
				vec := s.Q
				if v == 1 {
					// rotate K
					vec = s.KeyCache[loff+pos*kvDim : loff+pos*kvDim+kvDim]
				}
				v0 := vec[i]
				v1 := vec[i+1]
				vec[i] = v0*fcr - v1*fci
				vec[i+1] = v0*fci + v1*fcr
			}
		}

		// multi-head attention
		// s.Q is dim in size
		// s.Att is (n_heads, seq_len)
		// We'll fill s.Xb with the final attention result (dim)
		for i := range s.Xb {
			s.Xb[i] = 0
		}

		// for each head, compute attention over all timesteps 0..pos
		for h := 0; h < nHeads; h++ {
			qPtr := s.Q[h*headSize : h*headSize+headSize]
			attPtr := s.Att[h*int(p.SeqLen) : h*int(p.SeqLen)+int(p.SeqLen)]
			// fill att scores
			for t := 0; t <= pos; t++ {
				kPtr := s.KeyCache[loff+t*kvDim+(h/kvMul)*headSize:]
				kPtr = kPtr[:headSize]
				score := float32(0.0)
				for i := 0; i < headSize; i++ {
					score += qPtr[i] * kPtr[i]
				}
				score /= float32(math.Sqrt(float64(headSize)))
				attPtr[t] = score
			}
			softmax(attPtr, pos+1)
			// weighted sum of values
			xbHeadPtr := s.Xb[h*headSize : h*headSize+headSize]
			for t := 0; t <= pos; t++ {
				vPtr := s.ValueCache[loff+t*kvDim+(h/kvMul)*headSize:]
				vPtr = vPtr[:headSize]
				a := attPtr[t]
				for i := 0; i < headSize; i++ {
					xbHeadPtr[i] += a * vPtr[i]
				}
			}
		}

		// final matmul after attention
		matmul(s.Xb2, s.Xb, w.Wo[l*dim*dim:l*dim*dim+dim*dim], dim, dim)
		// residual
		for i := 0; i < dim; i++ {
			x[i] += s.Xb2[i]
		}

		// ffn rmsnorm
		rmsnorm(s.Xb, x, w.RmsFfnWeight[l*dim:l*dim+dim], dim)

		// W1, W3
		matmul(s.Hb, s.Xb, w.W1[l*dim*hiddenDim:l*dim*hiddenDim+dim*hiddenDim], dim, hiddenDim)
		matmul(s.Hb2, s.Xb, w.W3[l*dim*hiddenDim:l*dim*hiddenDim+dim*hiddenDim], dim, hiddenDim)

		// SwiGLU
		for i := 0; i < hiddenDim; i++ {
			val := s.Hb[i]
			// silu
			val = val * (1.0 / (1.0 + float32(math.Exp(float64(-val)))))
			val *= s.Hb2[i]
			s.Hb[i] = val
		}

		// W2
		matmul(s.Xb, s.Hb, w.W2[l*dim*hiddenDim:l*dim*hiddenDim+hiddenDim*dim], hiddenDim, dim)
		// residual
		for i := 0; i < dim; i++ {
			x[i] += s.Xb[i]
		}
	}

	// final rmsnorm
	rmsnorm(x, x, w.RmsFinalWeight, dim)
	// classifier
	matmul(s.Logits, x, w.Wcls, dim, int(p.VocabSize))

	return s.Logits
}

// ----------------------------------------------------------------------------
// Tokenizer: Byte Pair Encoding (BPE)-like, from the reference code

type TokenIndex struct {
	Str string
	Id  int
}

type Tokenizer struct {
	Vocab       []string
	VocabScores []float32
	SortedVocab []TokenIndex
	VocabSize   int
	MaxTokenLen int
	BytePieces  [512]byte // stores 256 2-byte strings (like <0xXX>\0)
}

// buildTokenizer loads a tokenizer binary and populates the Tokenizer
func buildTokenizer(t *Tokenizer, tokenizerPath string, vocabSize int32) error {
	f, err := os.Open(tokenizerPath)
	if err != nil {
		return fmt.Errorf("couldn't load tokenizer %s: %v", tokenizerPath, err)
	}
	defer f.Close()

	t.VocabSize = int(vocabSize)
	t.Vocab = make([]string, vocabSize)
	t.VocabScores = make([]float32, vocabSize)
	t.SortedVocab = nil

	for i := 0; i < 256; i++ {
		// store each single-byte string, e.g. for 0x01 => [0x01, 0x00]
		t.BytePieces[i*2] = byte(i)
		t.BytePieces[i*2+1] = 0
	}

	// first 4 bytes => t.MaxTokenLen
	var maxTokenLen int32
	if err := binary.Read(f, binary.LittleEndian, &maxTokenLen); err != nil {
		return fmt.Errorf("failed to read max_token_length: %v", err)
	}
	t.MaxTokenLen = int(maxTokenLen)

	// read each vocab item: float32, then length (int32), then string
	buf := make([]byte, 0, 256)
	for i := 0; i < int(vocabSize); i++ {
		if err := binary.Read(f, binary.LittleEndian, &t.VocabScores[i]); err != nil {
			return fmt.Errorf("failed to read vocab_scores: %v", err)
		}
		var length int32
		if err := binary.Read(f, binary.LittleEndian, &length); err != nil {
			return fmt.Errorf("failed to read length: %v", err)
		}
		if length < 0 || length > 65536 {
			return fmt.Errorf("invalid token length: %d", length)
		}
		buf = make([]byte, length)
		if _, err := io.ReadFull(f, buf); err != nil {
			return fmt.Errorf("failed read token string: %v", err)
		}
		t.Vocab[i] = string(buf)
	}
	return nil
}

func (t *Tokenizer) decode(prevToken, token int) string {
	piece := t.Vocab[token]
	// following BOS (1) token, strip leading space if present
	if prevToken == 1 && len(piece) > 0 && piece[0] == ' ' {
		piece = piece[1:]
	}

	// fix: check if piece is like "<0x0A>"
	if strings.HasPrefix(piece, "<0x") && len(piece) == 6 && piece[5] == '>' {
		var bVal byte
		_, err := fmt.Sscanf(piece, "<0x%02X>", &bVal)
		if err == nil {
			// return single raw byte
			return string(t.BytePieces[bVal*2 : bVal*2+1])
		}
	}
	return piece
}

func safePrintf(piece string) {
	// If it’s exactly one byte: check if it’s a newline or other control
	if len(piece) == 1 {
		b := piece[0]
		if b == '\n' {
			fmt.Println() // print a real newline
			return
		}
		// If it is “printable or whitespace,” go ahead
		if !unicode.IsPrint(rune(b)) && !unicode.IsSpace(rune(b)) {
			return
		}
	}
	fmt.Print(piece)
}

func (t *Tokenizer) ensureSortedVocab() {
	if t.SortedVocab != nil {
		return
	}
	t.SortedVocab = make([]TokenIndex, t.VocabSize)
	for i := 0; i < t.VocabSize; i++ {
		t.SortedVocab[i] = TokenIndex{
			Str: t.Vocab[i],
			Id:  i,
		}
	}
	sort.Slice(t.SortedVocab, func(i, j int) bool {
		return t.SortedVocab[i].Str < t.SortedVocab[j].Str
	})
}

func (t *Tokenizer) strLookup(str string) int {
	// binary search
	t.ensureSortedVocab()
	lo, hi := 0, t.VocabSize
	for lo < hi {
		mid := (lo + hi) >> 1
		if t.SortedVocab[mid].Str < str {
			lo = mid + 1
		} else if t.SortedVocab[mid].Str > str {
			hi = mid
		} else {
			return t.SortedVocab[mid].Id
		}
	}
	return -1
}

func (t *Tokenizer) encode(text string, bos, eos bool, tokens []int) (int, error) {
	// bos=1 => prepend BOS(=1), eos=1 => append EOS(=2)
	nTokens := 0
	if text == "" {
		return 0, errors.New("cannot encode empty string? (or do you want 0 tokens?)")
	}
	if bos {
		tokens[nTokens] = 1
		nTokens++
	}
	// add dummy prefix if text not empty
	if text != "" {
		spID := t.strLookup(" ")
		if spID < 0 {
			return nTokens, fmt.Errorf("couldn't find dummy prefix ' ' in vocab?")
		}
		tokens[nTokens] = spID
		nTokens++
	}

	// We'll parse UTF-8 codepoints from text
	// But the original code does partial merges, etc. We'll do a direct port.

	// var bufRunes []rune
	var codepoint string

	// read text as runes, but replicate the logic of splitting by codepoints and fallback to bytes
	textRunes := []rune(text)
	i := 0
	for i < len(textRunes) {
		// accumulate next codepoint as a single “unicode character” in codepoint
		r := textRunes[i]
		// in the reference C code, it tries to see if the codepoint exists in vocab
		codepoint = string(r)
		idx := t.strLookup(codepoint)
		if idx >= 0 {
			// we found it
			tokens[nTokens] = idx
			nTokens++
		} else {
			// fallback to bytes
			b := []byte(string(r))
			for _, bb := range b {
				tokens[nTokens] = int(bb) + 3 // +3 offset
				nTokens++
			}
		}
		i++
	}

	// merge consecutive pairs if found in vocab
mergeLoop:
	for {
		bestScore := float32(-1e10)
		bestID := -1
		bestIdx := -1
		for i := 0; i < (nTokens - 1); i++ {
			merged := t.Vocab[tokens[i]] + t.Vocab[tokens[i+1]]
			id := t.strLookup(merged)
			if id != -1 && t.VocabScores[id] > bestScore {
				bestScore = t.VocabScores[id]
				bestID = id
				bestIdx = i
			}
		}
		if bestIdx == -1 {
			break mergeLoop
		}
		// merge
		tokens[bestIdx] = bestID
		copy(tokens[bestIdx+1:], tokens[bestIdx+2:nTokens])
		nTokens--
	}

	if eos {
		tokens[nTokens] = 2
		nTokens++
	}
	return nTokens, nil
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

func sampleArgmax(probabilities []float32, n int) int {
	maxI := 0
	maxV := probabilities[0]
	for i := 1; i < n; i++ {
		if probabilities[i] > maxV {
			maxV = probabilities[i]
			maxI = i
		}
	}
	return maxI
}

func sampleMult(probabilities []float32, n int, coin float32) int {
	cdf := float32(0.0)
	for i := 0; i < n; i++ {
		cdf += probabilities[i]
		if coin < cdf {
			return i
		}
	}
	return n - 1
}

func (s *Sampler) sampleTopP(probabilities []float32, n int, coin float32) int {
	// gather subset that is >= some cutoff, then sort descending
	cutoff := (1.0 - s.TopP) / float32(n-1)
	n0 := 0
	for i := 0; i < n; i++ {
		if probabilities[i] >= cutoff {
			s.ProbIndex[n0].Index = i
			s.ProbIndex[n0].Prob = probabilities[i]
			n0++
		}
	}
	sort.Slice(s.ProbIndex[:n0], func(i, j int) bool {
		return s.ProbIndex[i].Prob > s.ProbIndex[j].Prob
	})
	// truncate
	cumProb := float32(0.0)
	lastIdx := n0 - 1
	for i := 0; i < n0; i++ {
		cumProb += s.ProbIndex[i].Prob
		if cumProb > s.TopP {
			lastIdx = i
			break
		}
	}
	r := coin * cumProb
	cdf := float32(0.0)
	for i := 0; i <= lastIdx; i++ {
		cdf += s.ProbIndex[i].Prob
		if r < cdf {
			return s.ProbIndex[i].Index
		}
	}
	return s.ProbIndex[lastIdx].Index
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
	coin := s.Rng.Float32() // random in [0,1)
	if s.TopP <= 0 || s.TopP >= 1 {
		return sampleMult(logits, s.VocabSize, coin)
	}
	return s.sampleTopP(logits, s.VocabSize, coin)
}

// ----------------------------------------------------------------------------
// Utilities

func timeInMs() int64 {
	return time.Now().UnixNano() / 1_000_000
}

// ----------------------------------------------------------------------------
// Generation

func generate(transformer *Transformer, tokenizer *Tokenizer, sampler *Sampler, prompt string, steps int) {
	// encode prompt
	// fmt.Printf("Debug - Prompt: '%s'\n", prompt)
	promptTokens := make([]int, len(prompt)+3) // upper-bound
	numPromptTokens, err := tokenizer.encode(prompt, true, false, promptTokens)
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode error: %v\n", err)
		return
	}
	if numPromptTokens < 1 {
		fmt.Fprintf(os.Stderr, "something is wrong, expected at least 1 prompt token\n")
		return
	}

	start := int64(0)
	var next int
	token := promptTokens[0]
	pos := 0
	for pos < steps {
		logits := forward(transformer, token, pos)
		if pos < (numPromptTokens - 1) {
			// force next prompt token
			next = promptTokens[pos+1]
		} else {
			next = sampler.sample(logits)
		}
		pos++
		if next == 1 { // BOS token
			break
		}
		piece := tokenizer.decode(token, next)
		safePrintf(piece)
		token = next
		if start == 0 {
			start = timeInMs()
		}
	}
	fmt.Println()
	if pos > 1 {
		end := timeInMs()
		tokS := float64(pos-1) / (float64(end-start) / 1000.0)
		fmt.Fprintf(os.Stderr, "achieved tok/s: %f\n", tokS)
	}
}

// ----------------------------------------------------------------------------
// Chat

func readStdin(guide string) string {
	fmt.Print(guide)
	scanner := bufio.NewScanner(os.Stdin)
	if !scanner.Scan() {
		return ""
	}
	return scanner.Text()
}

func chat(transformer *Transformer, tokenizer *Tokenizer, sampler *Sampler,
	cliUserPrompt, cliSystemPrompt string, steps int) {

	systemPromptBuf := ""
	userPromptBuf := ""
	renderedPrompt := ""

	promptTokens := make([]int, 1152)
	var userIdx int

	userTurn := true
	var next int
	var token int
	pos := 0

	for pos < steps {
		if userTurn {
			if pos == 0 {
				if cliSystemPrompt == "" {
					systemPromptBuf = readStdin("Enter system prompt (optional): ")
				} else {
					systemPromptBuf = cliSystemPrompt
				}
			}
			if pos == 0 && cliUserPrompt != "" {
				userPromptBuf = cliUserPrompt
			} else {
				userPromptBuf = readStdin("User: ")
			}

			// format as Llama2 Chat style
			if pos == 0 && systemPromptBuf != "" {
				systemTemplate := "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]"
				renderedPrompt = fmt.Sprintf(systemTemplate, systemPromptBuf, userPromptBuf)
			} else {
				userTemplate := "[INST] %s [/INST]"
				renderedPrompt = fmt.Sprintf(userTemplate, userPromptBuf)
			}

			nTok, err := tokenizer.encode(renderedPrompt, true, false, promptTokens)
			if err != nil {
				fmt.Fprintf(os.Stderr, "encode error: %v\n", err)
				return
			}
			userIdx = 0
			userTurn = false
			fmt.Print("Assistant: ")
			// we continue...
			fmt.Println(nTok)
		}

		if userIdx < len(promptTokens) && promptTokens[userIdx] != 0 {
			// still processing the input prompt tokens
			token = promptTokens[userIdx]
			userIdx++
		} else {
			// continue from the last sampled token
			token = next
		}
		// if token == 2 => EOS => user turn
		if token == 2 {
			fmt.Println()
			userTurn = true
		}
		logits := forward(transformer, token, pos)
		next = sampler.sample(logits)
		pos++

		// if we are out of the user prompt and the next token isn't EOS, print it
		if userIdx >= len(promptTokens) || promptTokens[userIdx] == 0 {
			if next != 2 {
				piece := tokenizer.decode(token, next)
				safePrintf(piece)
			} else {
				// next is EOS => done with assistant turn
			}
		}
	}
	fmt.Println()
}

// ----------------------------------------------------------------------------
// CLI main

func usage() {
	fmt.Fprintf(os.Stderr, "Usage:   %s <checkpoint> [options]\n", os.Args[0])
	fmt.Fprintf(os.Stderr, "Example: %s model.bin -n 256 -i \"Once upon a time\"\n", os.Args[0])
	fmt.Fprintf(os.Stderr, "Options:\n")
	fmt.Fprintf(os.Stderr, "  -t <float>  temperature in [0,inf], default 1.0\n")
	fmt.Fprintf(os.Stderr, "  -p <float>  p value in top-p sampling in [0,1], default 0.9\n")
	fmt.Fprintf(os.Stderr, "  -s <int>    random seed, default time.Now().Unix()\n")
	fmt.Fprintf(os.Stderr, "  -n <int>    number of steps, default 256, 0 => max_seq_len\n")
	fmt.Fprintf(os.Stderr, "  -i <string> input prompt\n")
	fmt.Fprintf(os.Stderr, "  -z <string> optional path to custom tokenizer\n")
	fmt.Fprintf(os.Stderr, "  -m <string> mode: generate|chat, default: generate\n")
	fmt.Fprintf(os.Stderr, "  -y <string> optional system prompt in chat mode\n")
}

var (
	temperature   = flag.Float64("t", 1.0, "temperature")
	topp          = flag.Float64("p", 0.9, "top-p")
	rngSeed       = flag.Int64("s", 0, "random seed")
	steps         = flag.Int("n", 256, "number of steps")
	promptStr     = flag.String("i", "", "input prompt")
	tokenizerPath = flag.String("z", "tokenizer.bin", "tokenizer path")
	mode          = flag.String("m", "generate", "mode: generate|chat")
	systemPrompt  = flag.String("y", "", "system prompt for chat")
)

func main() {
	// Set custom usage
	flag.Usage = usage
	flag.Parse()

	// Get positional arguments
	args := flag.Args()

	if len(args) < 1 {
		usage()
		os.Exit(1)
	}
	checkpointPath := args[0]

	// Validate parameters
	if *rngSeed <= 0 {
		*rngSeed = time.Now().Unix()
	}
	if *temperature < 0 {
		*temperature = 0
	}
	if *topp < 0.0 || *topp > 1.0 {
		*topp = 0.9
	}

	// Build the Transformer
	var t Transformer
	err := buildTransformer(&t, checkpointPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to build transformer: %v\n", err)
		os.Exit(1)
	}

	if *steps <= 0 || *steps > int(t.Config.SeqLen) {
		*steps = int(t.Config.SeqLen)
	}

	// Build the Tokenizer
	var tokenizer Tokenizer
	err = buildTokenizer(&tokenizer, *tokenizerPath, t.Config.VocabSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to build tokenizer: %v\n", err)
		os.Exit(1)
	}

	// Build Sampler
	sampler := buildSampler(int(t.Config.VocabSize), float32(*temperature), float32(*topp), *rngSeed)

	// run
	if *mode == "generate" {
		generate(&t, &tokenizer, &sampler, *promptStr, *steps)
	} else if *mode == "chat" {
		chat(&t, &tokenizer, &sampler, *promptStr, *systemPrompt, *steps)
	} else {
		fmt.Fprintf(os.Stderr, "unknown mode: %s\n", *mode)
		usage()
	}
}
