package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/victorddiniz/fitness-function-analysis/experiments"
)

func main() {
	experiment := experiments.NewExperiment(time.Now().UnixNano(), 10, 1, 1, 10000, "datasets/lynx.txt")
	results, _ := experiment.Run()

	for _, fitValues := range results {
		stringResults := make([]string, len(fitValues))
		for i, value := range fitValues {
			stringResults[i] = fmt.Sprintf("%f", value)
		}
		r := strings.Join(stringResults, ",")
		fmt.Println(r)
	}
}
