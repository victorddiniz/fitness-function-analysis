package functions

// F1 ...
func F1(t, o []float64) float64 {
	return 1.0 / (1.0 + ARV(t, o))
}

// F2 ...
func F2(t, o []float64) float64 {
	return 1.0 / (1.0 + MSE(t, o))
}

//F3 ...
func F3(t, o []float64) float64 {
	return 1.0 / (1.0 + Theil(t, o))
}

//F4 ...
func F4(t, o []float64) float64 {
	sumErrors := ARV(t, o) + MSE(t, o) + Theil(t, o) + MAPE(t, o)
	return POCID(t, o) / (1.0 + sumErrors)
}

//F5 ...
func F5(t, o []float64) float64 {
	return 1.0 / (1.0 + MAPE(t, o))
}

//F6 ...
func F6(t, o []float64) float64 {
	return POCID(t, o) / (1.0 + ARV(t, o))
}

//F7 ...
func F7(t, o []float64) float64 {
	return POCID(t, o) / (1.0 + Theil(t, o))
}

//F8 ...
func F8(t, o []float64) float64 {
	return POCID(t, o) / (1.0 + MSE(t, o))
}
