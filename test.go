package main

import (
	"fmt"
	"time"
	"github.com/victorddiniz/fitness-function-analysis/experiments"
)

func main() {
	experiment := experiments.NewExperiment(time.Now().UnixNano(), 10, 1, 1, 100000, "datasets/sun.txt")
	results := experiment.Run()
	
	fmt.Println(results)
}
