package main

import (
	"github.com/victorddiniz/fitness-function-analysis/functions"
	"fmt"
	"math/rand"
	"github.com/victorddiniz/fitness-function-analysis/iohandlers"
	"github.com/victorddiniz/fitness-function-analysis/es"
)

func main() {
	rand := rand.New(rand.NewSource(1))
	var bestInd * es.Individual
	bestValidation := 0.0

	for i := 0; i < 10; i++ {
		population := es.NewPopulation(1, 1, 100000, "../datasets/sun.txt", rand)
		ioHandler := iohandlers.GetInstance()
		indRun, valueTraining := population.Run()
		in, out := ioHandler.GetKLagValidationSet(indRun.GetLag())
		value := indRun.Fitness(in, out)

		if bestInd == nil || value > bestValidation {
			bestInd = indRun
			bestValidation = value
		}
		fmt.Println(value, valueTraining)
	}

	ioHandler := iohandlers.GetInstance()
	in, target := ioHandler.GetKLagTestSet(bestInd.GetLag())
	obs := bestInd.Predict(in)
	fmt.Println(functions.MSE(target, obs))
}
