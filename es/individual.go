package es

import (
	"time"
	"github.com/victorddiniz/fitness-function-analysis/functions"
	"math"
	"math/rand"
)

var randSource *rand.Rand

// Individual ...
type Individual struct {
	weights [][]float64
	mSteps [][]float64
	t float64
	tLine float64
	hidden int
	lag int
}

func (ind *Individual) epsilon() float64 {
	return 1e-3
}

func (ind *Individual) mutateMS(i, j int) {
	ms := ind.mSteps[i][j]
	t := ind.t
	tLine := ind.tLine

	ms *= math.Exp(tLine * randSource.NormFloat64() + t * randSource.NormFloat64())
	if ms < ind.epsilon() {
		ms = ind.epsilon()
	}
	ind.mSteps[i][j] = ms
}

func (ind *Individual) mutateWeight(i, j int) {
	ms := ind.mSteps[i][j]
	weight := ind.weights[i][j]

	ind.weights[i][j] = weight + ms * randSource.NormFloat64()
}

func (ind *Individual) predictSolo(input []float64) float64 {
	intermediarySumSig := 0.0
	for i := 0; i < ind.hidden; i++ {
		partial := 0.0
		for j := 0; j < ind.lag; j++ {
			partial += ind.weights[j][i] * input[j]
		}
		partial = functions.Sigmoid(partial)
		intermediarySumSig += partial * ind.weights[ind.lag][i]
	}
	return functions.Sigmoid(intermediarySumSig)

}

func (ind *Individual) copy() *Individual {
	var newInd Individual
	newInd.weights = make([][]float64, len(ind.weights))
	newInd.mSteps = make([][]float64, len(ind.mSteps))
	copy(newInd.weights, ind.weights)
	copy(newInd.mSteps, ind.mSteps)
	newInd.t, newInd.tLine = ind.t, ind.tLine
	newInd.hidden, newInd.lag = ind.hidden, ind.lag
	return &newInd
}

// GetWeights ...
func (ind *Individual) GetWeights() [][]float64 {
	return ind.weights
}

// GetMutationSteps ...
func (ind *Individual) GetMutationSteps() [][]float64 {
	return ind.mSteps
}

// GetLag ...
func (ind *Individual) GetLag() int {
	return ind.lag
}

// Fitness ...
func (ind *Individual) Fitness(input [][]float64, target []float64) float64 {
	output := ind.Predict(input)
	return 1.0/(1.0 + functions.MSE(target, output))
}

// Predict ...
func (ind *Individual) Predict(input [][]float64) []float64 {
	prediction := make([]float64, len(input))
	for index, value := range input {
		prediction[index] = ind.predictSolo(value)
	}
	return prediction
}

// Mutate ...
func (ind Individual) Mutate() *Individual{
	newInd := ind.copy()
	for i := 0; i <= newInd.lag; i++ {
		for j := 0; j < newInd.hidden; j++ {
			newInd.mutateMS(i, j)
			newInd.mutateWeight(i, j)
		}
	}
	return newInd
}

// NewIndividual ...
func NewIndividual(lag, hidden int) (*Individual) {
	weights := make([][]float64, lag + 1)
	mSteps := make([][]float64, lag + 1)
	wTotal := float64(lag * hidden)
	t := 1.0/(math.Sqrt(2.0 * math.Sqrt(wTotal)))
	tLine := 1.0/(math.Sqrt(2.0 * wTotal))
	seed := rand.NewSource(time.Now().UnixNano())
	randSource = rand.New(seed)

	for i := 0; i <= lag; i++ {
		weights[i] = make([]float64, hidden)
		mSteps[i] = make([]float64, hidden)
		for j := 0; j < hidden; j++ {
			weights[i][j] = math.Max(randSource.NormFloat64() * 0.5/3.0 + 0.5, 0.0)
			weights[i][j] = math.Min(weights[i][j], 1.0)
			mSteps[i][j] = randSource.Float64()
		}
	}

	return &Individual{
		weights: weights,
		mSteps: mSteps,
		t:t,
		tLine: tLine,
		hidden: hidden,
		lag: lag}
}