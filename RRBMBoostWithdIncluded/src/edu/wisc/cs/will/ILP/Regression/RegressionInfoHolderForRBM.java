package edu.wisc.cs.will.ILP.Regression;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.wisc.cs.will.Boosting.RDN.RegressionRDNExample;
import edu.wisc.cs.will.Boosting.Utils.BoostingUtils;
import edu.wisc.cs.will.DataSetUtils.Example;
import edu.wisc.cs.will.DataSetUtils.RegressionExample;
import edu.wisc.cs.will.ILP.LearnOneClause;
import edu.wisc.cs.will.ILP.SingleClauseNode;
import edu.wisc.cs.will.Utils.ProbDistribution;
import edu.wisc.cs.will.Utils.Utils;
import edu.wisc.cs.will.stdAIsearch.SearchInterrupted;

public class RegressionInfoHolderForRBM extends RegressionInfoHolderForRDN{
    
	double epsilon = 0.000025;
	//double epsilon = 0.5;
	double learningRate = 0.05;
	//double threshold = 5.0;
	
	public RegressionInfoHolderForRBM() {
		super();
	}
	
	@Override
	public void populateExamples(LearnOneClause task, SingleClauseNode caller) throws SearchInterrupted {
		if (!task.regressionTask) { Utils.error("Should call this when NOT doing regression."); }
		if (caller.getPosCoverage() < 0.0) { caller.computeCoverage(); }
		
		List<Double> TrueBranchGradients = new ArrayList<Double>(); //required for stochastic version of coordinate gradient descent
		List<Double> FalseBranchGradients = new ArrayList<Double>(); //required for stochastic version of coordinate gradient descent
		
		for (Example posEx : task.getPosExamples()) {
			double weight = posEx.getWeightOnExample();
			double output = ((RegressionExample) posEx).getOutputValue();
					
			ProbDistribution prob   = ((RegressionRDNExample)posEx).getProbOfExample();
			
			//Utils.println("Navdeep : pos example: "+posEx.toString()+" :output: "+output+" : prob: "+prob);
			if (prob.isHasDistribution()) {
				Utils.error("Expected single probability value but contains distribution");
			}
			if (!caller.posExampleAlreadyExcluded(posEx)) { // This checks for the examples that failed the clause.
				
				//Utils.println("Positive Examples "+posEx.toString()+" Gradient: "+output);
				TrueBranchGradients.add(output);
				trueStats.addNumOutput(1, output, weight, prob.getProbOfBeingTrue());		
			}
			else
			{
				FalseBranchGradients.add(output);
			}
		}
				
		RegressionInfoHolder totalFalseStats = caller.getTotalFalseBranchHolder() ;
		if (totalFalseStats != null) {
			falseStats = falseStats.add(((RegressionInfoHolderForRDN)totalFalseStats).falseStats);
		}
		
		//Utils.println(" True Examples: "+trueStats.getNumExamples()+"  False Examples :  "+falseStats.getNumExamples());
		
		if (!(trueStats.getNumExamples() < task.getMinPosCoverage())){
			Utils.println("Making a call to Coordinate Gradient Descent for learning the parameters of model for the true branch");	
			ComputeCoordinateDescent(TrueBranchGradients,trueStats.getSumOfOutputAndNumGrounding(), trueStats.getNumExamples(),true);
		}			
				
		if (!(falseStats.getNumExamples() < task.getMinPosCoverage())){
			Utils.println("Making a call to Coordinate Gradient Descent for learning the parameters of model for the false branch");	
			ComputeCoordinateDescent(FalseBranchGradients,falseStats.getSumOfOutputAndNumGrounding(), falseStats.getNumExamples(),false);	
		}
				
	}
	
	
	public void ComputeCoordinateDescent(List<Double> branchGradients, double SumDelta, double totalExamplesSatisfyingBranch, boolean trueBranch)
	{
		Utils.print("Inside Coordinate Gradient Descent");
		if(trueBranch)
			Utils.println(" for true branch");
		else
			Utils.println(" for false branch");
		
		Utils.println("");
		Utils.println("SumDelta is = "+SumDelta+" totalExamplesSatisfyingBranch the branch = "+totalExamplesSatisfyingBranch);
		
		double d  = 0;
		double c  = 0;
		double U0 = 0;
		double U1 = 0;
		double w  = 0;
		
		boolean convergence = false;
	    double t = 1;
		while(!convergence)
		{
			
			double Old_U0, Old_U1, Old_c, Old_w, Old_d;
			
			Old_d  = d;
			Old_U1 = U1;
			Old_U0 = U0;
			Old_c  = c;
			Old_w  = w;
			
			Collections.shuffle(branchGradients);
			
			/*Utils.println(" ");
			Utils.println("***** Iteration "+t+" of Coordinate Gradient Descent *****");
			Utils.println(" ");*/
			
			for(int exampleNumber = 0; exampleNumber < branchGradients.size(); exampleNumber++)
			{
				double Delta = branchGradients.get(exampleNumber);
			
				double GradientOfd = ComputePhiMinusDelta(c, U0, U1, w, d, Delta);
				d = d -  learningRate*GradientOfd;
				/*
				Utils.println("GradientOfd = "+GradientOfd+", d after updating= "+d);
				Utils.println(" ");*/
				
				double GradientOfU1 = ComputePhiMinusDelta(c, U0, U1, w, d, Delta)*BoostingUtils.sigmoid((c + U1 + w), 0);
				U1 = U1 -  learningRate*GradientOfU1;
				/*
				Utils.println("GradientOfU1 = "+GradientOfU1+", U1 after updating= "+U1);
				Utils.println(" ");*/
			
				double GradientOfU0 = ComputePhiMinusDelta(c, U0, U1, w, d, Delta)*(-BoostingUtils.sigmoid((c + U0 + w), 0));
				U0 = U0 -  learningRate*GradientOfU0;
				/*
				Utils.println("GradientOfU0 = "+GradientOfU0+", U0 after updating= "+U0);
				Utils.println(" ");*/
			
				double GradientOfw = ComputePhiMinusDelta(c, U0, U1, w, d, Delta)*(BoostingUtils.sigmoid((c + U1 + w),0) - BoostingUtils.sigmoid((c + U0 + w), 0));
				w = w -    learningRate*GradientOfw;
				
				/*Utils.println("GradientOfw = "+GradientOfw+", w after updating= "+w);
				Utils.println(" ");*/
			
				double GradientOfc = ComputePhiMinusDelta(c, U0, U1, w, d,  Delta)*(BoostingUtils.sigmoid((c + U1 + w),0) - BoostingUtils.sigmoid((c + U0 + w), 0));
				c = c -    learningRate*GradientOfc;
				/*Utils.println("GradientOfc = "+GradientOfc+", c after updating= "+c);
				Utils.println(" ");*/
			
			}
			//System.out.println("d = "+d+" U1= "+U1+" U0= "+U0+" c= "+c+" w= "+w);
			
			if((Math.pow((U1 - Old_U1), 2) + Math.pow((U0 - Old_U0), 2) + Math.pow((c - Old_c), 2) + Math.pow((w - Old_w), 2) + Math.pow((d - Old_d), 2)) <= epsilon)
			{
				convergence = true;
			}
			
			t = t + 1;
			learningRate = learningRate/Math.sqrt(t);
		}
		
		if(trueBranch)
		{
			trueStats.setParametersforRBM(c, U0, U1, w, d);
		}
		else
		{
			falseStats.setParametersforRBM(c, U0, U1, w, d);
		}
		
		Utils.println("parameters of RBM are c = "+Utils.truncate(c, 6)+", U0 = "+Utils.truncate(U0, 6)+", U1 = "+Utils.truncate(U1, 6)+", w = "+Utils.truncate(w,6)+", d = "+Utils.truncate(d,6));
		double phitemp = ComputePhi(c, U0, U1, w, d);
		Utils.println(" ");
		Utils.println("phi = "+phitemp+", (phi*totalExamplesSatisfyingBranch - SumDelta) = "+((phitemp*totalExamplesSatisfyingBranch)-SumDelta));
		Utils.println(" ");
	}
	
	public double ComputePhiMinusDelta(double c, double U0, double U1,double w, double d, double Delta)
	{
		return (ComputePhi(c, U0, U1, w, d)- Delta); 
	}
	
	public double ComputePhi(double c, double U0, double U1, double w, double d)
	{
		double phi  = d + Math.log((1 + Math.exp(c + U1 + w))/ (1 + Math.exp(c + U0 + w)));
		return phi;
	}
	

	/*public static double L2Norm(double U1,double U0, double w,double c, double d)
	{
		return (Math.sqrt(Math.pow(U1, 2) + Math.pow(U0, 2) + Math.pow(w, 2) + Math.pow(c, 2) + Math.pow(d, 2)));
		
	}
	*/
	public double getRBMCost(BranchStats stats) {
		
		double result = 0;
		
		double phi  = stats.getdforRBM() + (Math.log((1 + Math.exp(stats.getCforRBM() + stats.getU1forRBM() + stats.getWforRBM()))/ (1 + Math.exp(stats.getCforRBM() + stats.getU0forRBM() + stats.getWforRBM()))));
		
		// SSE calculation for a branch 0.5*sum_{x}(phi-delta)^2 expanded into its formula  = 0.5*phi*phi*|J| + 0.5*\sum_{x}delta_{x}*delta_{x} - phi\sum_{x}delta_{x}
		result = (0.5*Math.pow(phi, 2)*stats.getNumExamples()) + 0.5*stats.getSumOfOutputSquared() - phi*stats.getSumOfOutputAndNumGrounding();
		
		return result;
	}
	
	@Override
	public double weightedVarianceAtSuccess() {		
		return getRBMCost(trueStats);
	}
	
	@Override
	public double weightedVarianceAtFailure() {
		return getRBMCost(falseStats);
	}
		
}
