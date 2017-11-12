import java.io.*;
import java.util.*;
import java.util.regex.*;
import javax.swing.*;
import java.awt.*;


import weka.classifiers.bayes.*;
import weka.classifiers.Classifier;
import weka.core.*;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.*;
import weka.attributeSelection.*;
import weka.core.stopwords.Rainbow;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.gui.visualize.*;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.core.converters.ArffSaver;
import weka.core.stemmers.*;

import org.jsoup.*;

@SuppressWarnings("deprecation") 
public class filter {

    public static void main(String[] args) {
        if (args.length == 0) {
            usage();
        }

        //Training
        if (args[0].equals("-t")) {
            if (args.length == 1) {
                usage();
            }

            final File directory = new File(args[1]);
            ArrayList<ArrayList<String>> testData = new ArrayList<ArrayList<String>>();

            // Reads in the files, creates a list of elements, each of which is a list containg
            //      - Class ("spam" or "ham")
            //      - preprocessing(all of the text in the file)
            for (final File entry : directory.listFiles()){
                ArrayList<String> entryData = readTrainingData(entry);
                testData.add(entryData);
            }

            FastVector<Attribute> fvAttributes = getAttributesVector();

            // Build training set of Instances
            Instances trainingSet = new Instances("Relation", fvAttributes, 10);
            trainingSet.setClassIndex(0);

            for(ArrayList<String> dataPoint : testData){
                Instance i = buildAnInstance(dataPoint, fvAttributes);
                trainingSet.add(i);
            }

            // Use StringToWordVector filter to change data into a word vector

            try {
                Filter vectorise = getFilter();
                vectorise.setInputFormat(trainingSet);
                Instances filteredTrainingSet = Filter.useFilter(trainingSet, vectorise);
                filteredTrainingSet.setClassIndex(0);


                
                // Attribute selection                
                AttributeSelection selection = new AttributeSelection();
                GainRatioAttributeEval eval = new GainRatioAttributeEval();
                Ranker search = new Ranker();
                selection.setSearch(search);
                selection.setEvaluator(eval);
                selection.setInputFormat(filteredTrainingSet);

                // Multifilter
                MultiFilter multi = new MultiFilter();
                Filter[] filters = new Filter[2];
                filters[0] = vectorise;
                filters[1] = selection;
                multi.setFilters(filters);

                // Classifier
                FilteredClassifier classifier = new FilteredClassifier();
                NaiveBayesMultinomial base = new NaiveBayesMultinomial();
                classifier.setFilter(multi);
                classifier.setClassifier(base);

                classifier.buildClassifier(trainingSet);
                weka.core.SerializationHelper.write("classifier.model", classifier);

                Instances selectedTrainingSet = Filter.useFilter(filteredTrainingSet, selection);
                ArffSaver saver = new ArffSaver();
                saver.setInstances(selectedTrainingSet);
                saver.setFile(new File("training.arff"));
                saver.writeBatch();

                // Classifer evaluation
                //classifierEval(classifier,selectedData);
                classifierEval(classifier,trainingSet);
            } 
            catch (Exception e){
                System.out.println("Error after trying to build and test classifier");
                System.exit(0);
            }
        }
        else {
            try {
                Classifier classifier = (Classifier) weka.core.SerializationHelper.read("classifier.model");
                //System.out.println("Loaded classifier");
                //System.out.println("Argument: "+args[0]);

                File testFile = new File(args[0]);
                ArrayList<String> fileData = readTrainingData(testFile);
                
                //System.out.println("Data is " + fileData.get(0));
                //System.out.println("Processed subject line: " + fileData.get(1));
                //System.out.println("Message:");
                //System.out.println(fileData.get(2));

                
                FastVector<Attribute> fvAttributes = getAttributesVector();
                Instances trainingSet = new Instances("Relation", fvAttributes, 10);
                trainingSet.setClassIndex(0);
                Instance i = buildAnInstance(fileData, fvAttributes);
                i.setDataset(trainingSet);

                //System.out.println("Built instance");

                double result = classifier.classifyInstance(i);
                if (result > 0.5) {
                    System.out.println("ham");
                }
                else {
                    System.out.println("spam");
                }
                //System.out.println("Classifier says: " + classifier.classifyInstance(i));
            }
            catch (Exception e) {
                System.out.println("Could not find model. Please train using the -t flag");
                //System.out.println("Error classifying:");
                System.out.println(e);
            }
        }
    }

    public static void usage() {
        System.out.println("Usage:\n\t java filter [-t] filename");
        System.exit(0);
    }

    public static Filter getFilter() {
        StringToWordVector filter = new StringToWordVector();
        Rainbow stopwordsHandler = new Rainbow();
        SnowballStemmer stemmer = new SnowballStemmer();
        String[] options = new String[4];
        options[0] = "-C";
        options[1] = "-stopwords-handler";
        options[2] = "Rainbow";
        options[3] = "-T";

        try {
            filter.setOptions(options);
        }
        catch (Exception e) {
            System.out.println("Could not set options for filter");
        }
        filter.setStopwordsHandler(stopwordsHandler);
        filter.setStemmer(stemmer);

        return filter;
    }


    public static FastVector<Attribute> getAttributesVector() {
        // Define Attributes
        // Class
        FastVector<String> fvClassVal = new FastVector<String>(2);
        fvClassVal.addElement("spam");
        fvClassVal.addElement("ham");
        Attribute classAttribute = new Attribute("theClass", fvClassVal);

        // Data
        Attribute data = new Attribute("data", (FastVector<String>) null);
        // Subject
        Attribute subject = new Attribute("subject", (FastVector<String>) null);

        // Declare a feature vector
        FastVector<Attribute> fvAttributes = new FastVector<Attribute>(3);
        fvAttributes.addElement(classAttribute);
        fvAttributes.addElement(subject);
        fvAttributes.addElement(data);

        return fvAttributes;
    }

    // Read a file to a String list {class, data} (File -> {class, data})
    public static ArrayList<String> readTrainingData(final File entry){
        BufferedReader reader;
        ArrayList<String> dataPoint = new ArrayList<String>();
        String data = "";

        try {
            reader = new BufferedReader(new FileReader(entry));
            
            String line;
                
            while ((line = reader.readLine()) != null) {
                data = data.concat(line + "\n");
            }
                
        }
        catch (FileNotFoundException e) {
            System.out.println("Could not find file " + entry);
            System.exit(0);
        }
        catch (IOException e) {
            System.out.println("Count not read file " + entry);
            System.exit(0);
        }

        String filename = entry.getName().toString();
        if (filename.charAt(0) == 's'){
            dataPoint.add("spam");
        } else {
            dataPoint.add("ham");
        }

        String[] processedData = preprocessing(data);
        dataPoint.add(processedData[0]);
        dataPoint.add(processedData[1]);

        return dataPoint;
    }

    // Preprocess data (String -> String)
    public static String[] preprocessing(String data){
        String[] out = new String[2];
        String processedData = "";
        String subject = "";
        
        try{
            BufferedReader reader = new BufferedReader(new StringReader(data));
            String line;
                    
            boolean body = false;
            int count = 0;
            String message = "";
            while ((line = reader.readLine()) != null) {
                count++;
                //System.out.println("Line "+count+" is empty? "+line.isEmpty());

                if (body) {
                    message =  message.concat(line + "\n");
                }
                else if (line.startsWith("Subject: ")) {
                    subject = line.substring("Subject: ".length());
                }
                
                if (!body && line.isEmpty()) {
                    body = true;
                }
            }
            
            /*String regex = "<.*?>";
            Pattern pattern = Pattern.compile(regex);
            Matcher matcher = pattern.matcher(message);
            
            while (matcher.find()) {
                for (int i = 1; i <= matcher.groupCount(); i++) {
                    System.out.println("Group " + i + ": " + matcher.group(i));
                }
            }*/
            
            //processedData = message.replaceAll("<!--(.*?)-->", " ");
            
            //Parses the HTML, finds all script and style tags, removes them, and gets the text from everything remaining
            /*org.jsoup.nodes.Document doc = Jsoup.parse(message);
            doc.select("script, style").remove();
            
            processedData = doc.text();*/
            processedData = Jsoup.parse(message).text();
            processedData = processedData.replaceAll("[^a-zA-Z0-9£$#\\/]", " ");
            processedData = processedData.replaceAll("/"," ");
        }
        catch(IOException e){
            System.out.println("Error preprocessing data");
            System.exit(0);
        }
        
        out[0] = subject;
        out[1] = processedData;
        return out;
    }

    // Build an Instance(class, data) for an ArrayList<String>[class, data] ({class,data} -> AttributeList -> Instance)
    public static Instance buildAnInstance(ArrayList<String> dataPoint, FastVector<Attribute> attributes){
        //System.out.println(attributes);
        //System.out.println(dataPoint);
        Instance instance = new DenseInstance(3);
        instance.setValue(attributes.elementAt(0),dataPoint.get(0));
        instance.setValue(attributes.elementAt(1),dataPoint.get(1));
        instance.setValue(attributes.elementAt(2),dataPoint.get(2));

        return instance;
    }

    // Classifier evaluation (print confusion matrix and do cross validation)
    public static void classifierEval(Classifier classifier, Instances trainingSet){
        try {
            System.out.println(((FilteredClassifier)classifier).globalInfo());
            Evaluation test = new Evaluation(trainingSet);
            Random rand = new Random(1);  // using seed = 1
            int folds = 10;
            test.crossValidateModel(classifier, trainingSet, folds, rand);
            System.out.println(test.toSummaryString());
            System.out.println(test.toMatrixString());
            System.out.println("tpr: " + test.truePositiveRate(0));
            System.out.println("tnr: " + test.trueNegativeRate(0));
            double avg_rec = (test.truePositiveRate(0) + test.trueNegativeRate(0))/2;
            System.out.println("Average recall: " + avg_rec*100 + "%");
            double weighted_rec = 0.86*test.truePositiveRate(0) + 0.14*test.trueNegativeRate(0);
            System.out.println("IRL 86% of emails are spam");
            System.out.println("Recall weighted as such: " + weighted_rec);

            // generate ROC curve
            ThresholdCurve tc = new ThresholdCurve();
            Instances result = tc.getCurve(test.predictions(), 0);
            ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
            vmc.setROCString("(Area under ROC = " + Utils.doubleToString(tc.getROCArea(result), 4) + ")");
            vmc.setName(result.relationName());
            PlotData2D tempd = new PlotData2D(result);
            tempd.setPlotName(result.relationName());
            tempd.addInstanceNumberAttribute();
            // specify which points are connected
            boolean[] cp = new boolean[result.numInstances()];
            for (int n = 1; n < cp.length; n++)
              cp[n] = true;
            tempd.setConnectPoints(cp);
            // add plot
            vmc.addPlot(tempd);
            // display curve
            String plotName = vmc.getName();
            final javax.swing.JFrame jf =
              new javax.swing.JFrame("Weka Classifier Visualize: "+plotName);
            jf.setSize(500,400);
            jf.getContentPane().setLayout(new BorderLayout());
            jf.getContentPane().add(vmc, BorderLayout.CENTER);
            jf.addWindowListener(new java.awt.event.WindowAdapter() {
              public void windowClosing(java.awt.event.WindowEvent e) {
                jf.dispose();
              }
            });
            jf.setVisible(true);
        }
        catch (Exception e) {
                System.out.println("Error evaluating classifer");
                System.exit(0);
        }
    }
}