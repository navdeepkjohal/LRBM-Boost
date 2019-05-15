package edu.wisc.cs.will.ILP;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import edu.wisc.cs.will.Boosting.RDN.RegressionRDNExample;
import edu.wisc.cs.will.DataSetUtils.Example;
import edu.wisc.cs.will.DataSetUtils.RegressionExample;
import edu.wisc.cs.will.FOPC.TreeStructuredTheoryInteriorNode;
import edu.wisc.cs.will.ILP.Regression.BranchStats;
import edu.wisc.cs.will.ILP.Regression.RegressionInfoHolderForRBM;
import edu.wisc.cs.will.Utils.Utils;
import edu.wisc.cs.will.Utils.VectorStatistics;

/**
 * @author shavlik
 *
 *
 *  Holds a task for learning an interior node for a tree-structured theory.
 *
 */
public class TreeStructuredLearningTask implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private List<Example>  posExamples = null;
	private List<Example>  negExamples = null;
	
	private TreeStructuredTheoryInteriorNode node;
	private SingleClauseNode          creatingNode;
	private double                    score;

	public TreeStructuredLearningTask(List<Example> posExamples, List<Example> negExamples, TreeStructuredTheoryInteriorNode node) {
		this.posExamples = posExamples;
		this.negExamples = negExamples;
		this.node        = node;
	}
	
	public List<Example> getPosExamples() {
		return posExamples;
	}

	public void setPosExamples(List<Example> posExamples) {
		this.posExamples = posExamples;
	}

	public List<Example> getNegExamples() {
		return negExamples;
	}

	public void setNegExamples(List<Example> negExamples) {
		this.negExamples = negExamples;
	}

	public TreeStructuredTheoryInteriorNode getNode() {
		return node;
	}

	public void setNode(TreeStructuredTheoryInteriorNode node) {
		this.node = node;
	}
	
	public SingleClauseNode getCreatingNode() {
		return creatingNode;
	}
	
	public void setCreatingNode(SingleClauseNode creatingNode) {
		this.creatingNode = creatingNode;
	}

	public double getScore() {
		return score;
	}

	public void setScore(double score) {
		this.score = score;
	}
	
	// This should be called ONLY for root nodes as SingleClauseNode object is not 
	// available(null) in that case. It assumes that the examples are regression examples 
	public double getVariance() {
		double sumOfOutputSquared = 0;
		double sumOfOutput = 0;
		double sumOfWeights = 0;
		for (Example eg : getPosExamples()) {
			RegressionExample regEx = (RegressionExample)eg;
			// If regression example, use vectorVariance
			if (regEx.isHasRegressionVector()) {
				return getVectorVariance();
			}
			sumOfOutputSquared += regEx.getOutputValue() * regEx.getOutputValue() * regEx.getWeightOnExample();
			sumOfOutput += regEx.getOutputValue() * regEx.getWeightOnExample();
			sumOfWeights += regEx.getWeightOnExample();
		}
		
        double variance =	sumOfOutputSquared/sumOfWeights - (Math.pow(sumOfOutput/sumOfWeights, 2));  
		return variance;
	}
	
	// This should be called ONLY for root nodes of RRBM-Boost. It calls Coordinate descent on entire data 
	// that is required to compute Phi for entire data.    This Phi is used to compute SSE for entire data.
	public double getVarianceForRBM()
	{	
		double variance  = 0;
		
		RegressionInfoHolderForRBM rbmInfo = new RegressionInfoHolderForRBM();
		List<Double> BranchGradients = new ArrayList<Double>();
		
		for (Example eg : getPosExamples()) {
			
			RegressionExample regEx = (RegressionExample)eg;
			double output = ((RegressionExample) regEx).getOutputValue();
			BranchGradients.add(output);
		
			rbmInfo.getTrueStats().addNumOutput(1, regEx.getOutputValue(), regEx.getWeightOnExample(), 0); // Since we are not using Prob of example anywhere, we set it to zero in last argument
																										   // ideally it should be obtained by ((RegressionRDNExample)regEx).getProbOfExample().getProbOfBeingTrue()
		}
				
		rbmInfo.ComputeCoordinateDescent(BranchGradients, rbmInfo.getTrueStats().getSumOfOutputAndNumGrounding(), rbmInfo.getTrueStats().getNumExamples(), true);
		variance = (rbmInfo.getRBMCost(rbmInfo.getTrueStats()))/(rbmInfo.getTrueStats().getNumExamples()); 
		
		return variance;
	}
	
	public double getVectorVariance() {
		VectorStatistics vecStats = new VectorStatistics();
		if (getPosExamples().size() == 0) {
			Utils.error("No examples in the task!!");
		}
		for (Example eg : getPosExamples()) {
			RegressionExample regEx = (RegressionExample)eg;
			if (regEx.isHasRegressionVector()) {
				vecStats.addVector(regEx.getOutputVector());
			} else {
				Utils.error("No regression vector for example: " + regEx.toString());
			}
		}
		return vecStats.getVariance();
	}
}

