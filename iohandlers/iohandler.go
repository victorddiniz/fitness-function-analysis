package iohandlers

import (
	"bufio"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

// IOHandler ...
type IOHandler struct {
	trainingSetByLag   [][][]float64
	validationSetByLag [][][]float64
	testSetByLag       [][][]float64
}

// SplitIO ...
func splitIO(lagK int, set [][][]float64) ([][]float64, []float64) {
	data := set[lagK-1]
	dataLen := len(data)
	in := make([][]float64, dataLen)
	out := make([]float64, dataLen)

	for index, value := range data {
		in[index] = value[:lagK]
		out[index] = value[lagK]
	}

	return in, out
}

// GetKLagTrainingSet ...
func (ioHandler *IOHandler) GetKLagTrainingSet(lagK int) ([][]float64, []float64) {
	return splitIO(lagK, ioHandler.trainingSetByLag)
}

// GetKLagValidationSet ...
func (ioHandler *IOHandler) GetKLagValidationSet(lagK int) ([][]float64, []float64) {
	return splitIO(lagK, ioHandler.validationSetByLag)
}

// GetKLagTestSet ...
func (ioHandler *IOHandler) GetKLagTestSet(lagK int) ([][]float64, []float64) {
	return splitIO(lagK, ioHandler.testSetByLag)
}


var instance *IOHandler

func splitDataInKLag(rawData []float64, lag int) [][]float64 {
	dataLen := len(rawData)
	dataByLag := make([][]float64, dataLen-lag)
	for i := 0; i < dataLen-lag; i++ {
		dataSlice := make([]float64, lag+1)
		for j := 0; j <= lag; j++ {
			dataSlice[j] = rawData[i+j]
		}
		dataByLag[i] = dataSlice
	}
	return dataByLag
}

// GetInstance ...
func GetInstance() *IOHandler {
	return instance
}

func splitInSets(rawData []float64) ([]float64, []float64, []float64) {
	totalNum := len(rawData)
	trainingOffset := 0
	validationOffset := totalNum / 2
	testOffset := validationOffset + totalNum/4
	trainingSet := make([]float64, validationOffset)
	validationSet := make([]float64, testOffset-validationOffset)
	testSet := make([]float64, totalNum-testOffset)

	for i := trainingOffset; i < validationOffset; i++ {
		trainingSet[i] = rawData[i]
	}
	for i := validationOffset; i < testOffset; i++ {
		validationSet[i-validationOffset] = rawData[i]
	}
	for i := testOffset; i < totalNum; i++ {
		testSet[i-testOffset] = rawData[i]
	}

	return trainingSet, validationSet, testSet
}

// NewIOHandler ...
func NewIOHandler(path string, maxLag int) *IOHandler {

	if instance == nil {
		fileObj, _ := os.Open(path)
		reader := bufio.NewReader(fileObj)
		var rawData []float64
		maxVal := 0.0
		minVal := 1000000000.0

		for {
			line, err := reader.ReadString('\n')
			trimmedNum := strings.TrimSpace(line)
			num, _ := strconv.ParseFloat(trimmedNum, 64)
			rawData = append(rawData, num)

			maxVal = math.Max(maxVal, num)
			minVal = math.Min(minVal, num)

			if err == io.EOF {
				break
			}
		}
		for index, value := range rawData {
			rawData[index] = (value - minVal) / (maxVal - minVal)
		}

		trainingSet, validationSet, testSet := splitInSets(rawData)

		trainingSetByLag := make([][][]float64, maxLag)
		validationSetByLag := make([][][]float64, maxLag)
		testSetByLag := make([][][]float64, maxLag)

		for i := 1; i <= maxLag; i++ {
			trainingSetByLag[i-1] = splitDataInKLag(trainingSet, i)
		}
		for i := 1; i <= maxLag; i++ {
			validationSetByLag[i-1] = splitDataInKLag(validationSet, i)
		}
		for i := 1; i <= maxLag; i++ {
			testSetByLag[i-1] = splitDataInKLag(testSet, i)
		}

		instance = &IOHandler{
			trainingSetByLag:   trainingSetByLag,
			validationSetByLag: validationSetByLag,
			testSetByLag:       testSetByLag}
	}

	return instance
}
