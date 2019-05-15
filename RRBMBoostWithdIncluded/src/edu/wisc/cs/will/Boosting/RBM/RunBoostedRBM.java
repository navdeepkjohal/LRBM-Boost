package edu.wisc.cs.will.Boosting.RBM;

import java.io.File;
import java.io.IOException;

import edu.wisc.cs.will.Boosting.Common.RunBoostedModels;
import edu.wisc.cs.will.Boosting.MLN.RunBoostedMLN;
import edu.wisc.cs.will.Boosting.RDN.RunBoostedRDN;
import edu.wisc.cs.will.Boosting.Utils.CommandLineArguments;
import edu.wisc.cs.will.Utils.Utils;
import edu.wisc.cs.will.Utils.check_disc;
import edu.wisc.cs.will.Utils.disc;


public class RunBoostedRBM extends RunBoostedRDN{

	public void runJob() {
		if (cmdArgs.isLearnVal()) {
			Utils.println("\n% Starting a LEARNING run of bRBM.");
			long start = System.currentTimeMillis();
			learnModel();
			long end = System.currentTimeMillis();
			Utils.println("\n% Total learning time ("  + Utils.comma(cmdArgs.getMaxTreesVal()) + " trees): " + Utils.convertMillisecondsToTimeSpan(end - start, 3) + ".");
		}
		if (cmdArgs.isInferVal()) {
			Utils.println("\n% Starting an INFERENCE run of bRBM.");
			long   start    = System.currentTimeMillis();
			inferModel();
			long end = System.currentTimeMillis();
			Utils.println("\n% Total inference time (" + Utils.comma(cmdArgs.getMaxTreesVal()) + " trees): " + Utils.convertMillisecondsToTimeSpan(end - start, 3) + ".");
		}
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		args = Utils.chopCommentFromArgs(args); 
		boolean disc_flag=false;
		CommandLineArguments cmd = RunBoostedModels.parseArgs(args);
		if (cmd == null) {
			Utils.error(CommandLineArguments.getUsageString());
		}
		disc discObj= new disc();
		
		/*Check for discretization*/
		
		check_disc flagObj=new check_disc();
		if (cmd.getTrainDirVal()!=null)
		{
			try {
			System.out.println("cmd.getTrainDirVal()"+cmd.getTrainDirVal());
			disc_flag=flagObj.checkflagvalues(cmd.getTrainDirVal());
			
			/*Updates the names of the training and Test file based on discretization is needed or not*/
			cmd.update_file_name(disc_flag);
//			System.out.println("Hellooooooooooooooooooooo"+cmd.get_filename());
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		}
		else if((cmd.getTestDirVal()!=null)) 
		{
			try {
			System.out.println("cmd.getTestDirVal()"+cmd.getTestDirVal());
			disc_flag=flagObj.checkflagvalues(cmd.getTestDirVal());
			
			/*Updates the names of the training and Test file based on discretization is needed or not*/
			cmd.update_file_name(disc_flag);
//			System.out.println("Hellooooooooooooooooooooo"+cmd.get_filename());
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
			
		}
		if (cmd.getTrainDirVal()!=null)
			{   
				File  f = new File(cmd.getTrainDirVal()+"\\"+cmd.trainDir+"_facts_disc.txt");
			    
				if(f.exists())
				 {
					f.delete();
				 }
				
			    try {
//			    	System.out.println("Hellooooooooooooooooooooo"+cmd.getTrainDirVal());
			    	if (disc_flag==true)
			    	{	
					discObj.Discretization(cmd.getTrainDirVal());
			    	}
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			    
			}
		if (cmd.getTestDirVal()!=null)
				
			{   
					
				File f = new File(cmd.getTestDirVal().replace("/","\\"+cmd.testDir+"_facts_disc.txt"));
				
				if(f.exists())
				{
					f.delete();
				}
				
				/*This module does the actual discretization step*/
			    try {
			    	if (disc_flag==true)
			    	{	
					 discObj.Discretization(cmd.getTestDirVal());
			    	} 
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			   
			}
		RunBoostedRBM runClass = null;
		runClass = new RunBoostedRBM();
		if (!cmd.isLearnRBM()) {
			Utils.waitHere("Set \"-rbm\"  in cmdline arguments to ensure that we intend to Learn RBM. Will now learn an RBM.");
		}
		cmd.setLearnRBM(true);
		
		runClass.setCmdArgs(cmd);
		runClass.runJob();
	}
		
}
