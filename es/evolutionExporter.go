package es

import (
	"encoding/csv"
	"fmt"
	"os"

	"github.com/victorddiniz/fitness-function-analysis/functions"
)

//EvolutionExporter ...
type EvolutionExporter struct {
	metricFile *csv.Writer
	fileObject *os.File
}

//WriteEvolution ...
func (ee *EvolutionExporter) WriteEvolution(target, observations []float64, fit float64) error {
	stringValues := []string{
		fmt.Sprintf("%f", functions.MSE(target, observations)),
		fmt.Sprintf("%f", functions.MAPE(target, observations)),
		fmt.Sprintf("%f", functions.Theil(target, observations)),
		fmt.Sprintf("%f", functions.POCID(target, observations)),
		fmt.Sprintf("%f", functions.ARV(target, observations)),
		fmt.Sprintf("%f", fit),
	}
	return ee.metricFile.Write(stringValues)
}

//Close ...
func (ee *EvolutionExporter) Close() {
	ee.metricFile.Flush()
	ee.fileObject.Close()
}

//NewEvolutionExporter ...
func NewEvolutionExporter(filePath string) (*EvolutionExporter, error) {
	fileObj, err := os.Create(filePath)
	if err != nil {
		return nil, err
	}
	csvFileWrite := csv.NewWriter(fileObj)
	err = csvFileWrite.Write([]string{"MSE", "MAPE", "THEIL", "POCID", "ARV", "FIT"})
	if err != nil {
		return nil, err
	}

	return &EvolutionExporter{
		metricFile: csvFileWrite,
		fileObject: fileObj,
	}, nil
}
