package edu.wisc.cs.will.ILP.Regression;

import edu.wisc.cs.will.Utils.Utils;

public class BranchStats {
	protected double sumOfOutputSquared = 0;
	//private double sumOfOutput = 0;
	//private double sumOfNumGrounding = 0;
	//private double weightedProb = 0;
	protected double sumOfNumGroundingSquared = 0;
	protected double sumOfNumGroundingSquaredWithProb = 0;
	protected double sumOfOutputAndNumGrounding = 0;
	protected double numExamples 	=	0;
	
	private double useFixedLambda = Double.NaN;
	
	
	// Not used but useful for debugging
	double numNegativeOutputs = 0;
	double numPositiveOutputs = 0;
	
	/////// five parameters of RBM /////////
	double c =  0.0;
	double U0 = 0.0;
	double U1 = 0.0;
	double W  = 0.0;
	double d  = 0.0;
	/////// parameters of RBM ends ////////
	
	
	public void addNumOutput(long num, double output, double weight,double prob) {
		 double deno   = prob * (1-prob);
         if (deno < 0.1) {
         	deno = 0.1; 
         }
      //  sumOfNumGrounding += num;
		sumOfNumGroundingSquared += num*num*weight;
      //  sumOfOutput += output;
        sumOfOutputAndNumGrounding += num*output*weight;
        sumOfOutputSquared += output * output*weight;
        if (output > 0 ) {
        	numPositiveOutputs+=weight; 
        } else {
        	numNegativeOutputs+=weight;
        }
        numExamples+=weight;
        sumOfNumGroundingSquaredWithProb = num*num*weight*deno;
	}
	public BranchStats add(BranchStats other) {
		BranchStats newStats = new BranchStats();
		addTo(other, newStats);
		return newStats;
	}
	
	public void addTo(BranchStats other, BranchStats newStats) {
		//newStats.sumOfNumGrounding = this.sumOfNumGrounding + other.sumOfNumGrounding;
		newStats.sumOfNumGroundingSquared = this.sumOfNumGroundingSquared + other.sumOfNumGroundingSquared;
		newStats.sumOfOutputAndNumGrounding = this.sumOfOutputAndNumGrounding + other.sumOfOutputAndNumGrounding;
		//newStats.sumOfOutput = this.sumOfOutput + other.sumOfOutput;
		newStats.sumOfOutputSquared = this.sumOfOutputSquared + other.sumOfOutputSquared;
		newStats.numNegativeOutputs = this.numNegativeOutputs + other.numNegativeOutputs;
		newStats.numPositiveOutputs = this.numPositiveOutputs + other.numPositiveOutputs;
		newStats.numExamples = this.numExamples + other.numExamples;
		newStats.sumOfNumGroundingSquaredWithProb = this.sumOfNumGroundingSquaredWithProb + other.sumOfNumGroundingSquaredWithProb;
		if (!Double.isNaN(this.useFixedLambda) || !Double.isNaN(other.useFixedLambda)) {
			if (this.useFixedLambda != other.useFixedLambda) {
				Utils.waitHere("Different lambdas for " + this.useFixedLambda + " & " + other.useFixedLambda);
			}	else {
				newStats.useFixedLambda = this.useFixedLambda;
			}
		}
	}
	public double getLambda() {
		return getLambda(false);
	}
	public double getLambda(boolean useProbWeights) {
	
		if (!Double.isNaN(useFixedLambda)) {
			return useFixedLambda;
		}
		if (sumOfNumGroundingSquared == 0) {
			return 0;
		}
		if (sumOfNumGroundingSquaredWithProb == 0) {
			Utils.waitHere("Groundings squared with prob is 0??");
		}
		double lambda =  sumOfOutputAndNumGrounding / sumOfNumGroundingSquared;
		if (useProbWeights) {
			//Utils.waitHere("Computations not correct for vector-based probabilities");
			lambda = sumOfOutputAndNumGrounding / sumOfNumGroundingSquaredWithProb;
		}
		
		//if (lambda == 0) {
		//	Utils.println(this.toAttrString());
		//}
		return lambda;
	}
	
	public double getWeightedVariance() {
		if (sumOfNumGroundingSquared == 0) {
			return 0;
		}
		//System.out.println(sumOfOutputSquared - (Math.pow(sumOfOutputAndNumGrounding, 2)/sumOfNumGroundingSquared));
		return sumOfOutputSquared - (Math.pow(sumOfOutputAndNumGrounding, 2)/sumOfNumGroundingSquared); 

	}
	
	public String toAttrString() {
		return 	"% Sum of Output squared		=	" + sumOfOutputSquared + "\n" +
		//"% Sum of Output 				=	" + sumOfOutput + "\n" +
		"% Sum of #groundings squared	=	" + sumOfNumGroundingSquared + "\n" +
		"% Sum of #groundings^2*Probs	=	" + sumOfNumGroundingSquaredWithProb + "\n" +
		//"% Sum of #groundings 			=	" + sumOfNumGrounding + "\n" +
		"% Sum of #groundings*output	=	" + sumOfOutputAndNumGrounding + "\n" +
		"% Num of +ve output			=	" + numPositiveOutputs + "\n" +
		"% Num of -ve output			=	" + numNegativeOutputs +  "\n" +
		"% parameter d  of RBM          =   " + d                  +  "\n"+
		"% parameter c  of RBM          =   " + c                  +  "\n"+
		"% parameter U0 of RBM          =   " + U0                 +  "\n"+
		"% parameter U1 of RBM          =   " + U1                 +  "\n"+
		"% parameter W of RBM           =   " + W                  +  "\n";
	}
	public String toString() {
		return toAttrString() + "\n" + 
				(!Double.isNaN(useFixedLambda) ?
				"% Fixed Lambda					=	" + useFixedLambda + "\n":"") +
				"% Lambda						=	" + getLambda()+ "\n" + 
				"% Prob Lambda					=	" + getLambda(true) ;
	}
	
	public void setZeroLambda() {
		useFixedLambda = 0;
	}
	/**
	 * @return the sumOfOutputSquared
	 */
	public double getSumOfOutputSquared() {
		return sumOfOutputSquared;
	}
	/**
	 * @return the sumOfNumGroundingSquared
	 */
	public double getSumOfNumGroundingSquared() {
		return sumOfNumGroundingSquared; 
	}
	/**
	 * @return the sumOfNumGroundingSquaredWithProb
	 */
	public double getSumOfNumGroundingSquaredWithProb() {
		return sumOfNumGroundingSquaredWithProb;
	}
	/**
	 * @return the sumOfOutputAndNumGrounding
	 */
	public double getSumOfOutputAndNumGrounding() {
		return sumOfOutputAndNumGrounding;
	}
	/**
	 * @return the numExamples
	 */
	public double getNumExamples() {
		return numExamples;
	}
	/**
	 * @return the useFixedLambda
	 */
	public double getUseFixedLambda() {
		return useFixedLambda;
	}
	/**
	 * @return the numNegativeOutputs
	 */
	public double getNumNegativeOutputs() {
		return numNegativeOutputs;
	}
	/**
	 * @return the numPositiveOutputs
	 */
	public double getNumPositiveOutputs() {
		return numPositiveOutputs;
	}
	
	public void setParametersforRBM(double cNew, double U0New, double U1New, double wNew, double dNew) // Added by Navdeep Kaur for RRBM-Boost
	{		c  = cNew;
			U0 = U0New;
			U1 = U1New;
			W  = wNew;
			d  = dNew;
	}
	
	public double getCforRBM() { // Added by Navdeep Kaur for RRBM-Boost
		return c;
	}
	
	public double getU0forRBM() { // Added by Navdeep Kaur for RRBM-Boost
		return U0;
	}
	public double getU1forRBM() { // Added by Navdeep Kaur for RRBM-Boost
		return U1;
	}
	public double getWforRBM() { // Added by Navdeep Kaur for RRBM-Boost
		return W;
	}
	public double getdforRBM() { // Added by Navdeep Kaur for RRBM-Boost
		return d;
	}


}

