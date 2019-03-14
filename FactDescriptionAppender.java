import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.naturalli.NaturalLogicAnnotations;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.PropertiesUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;

public class FactDescriptionAppender {

    private FactDescriptionAppender() {}

    private static <T> boolean containsInOrder(List<T> list, List<T> sublist) {
        if (list.size() < sublist.size()) {
            return false;
        }
        Iterator<T> iter = list.iterator();
        for (T item : sublist) {
            boolean found = false;
            while (iter.hasNext()) {
                if (iter.next().equals(item)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
        return true;
    }

    private static <T> List<List<T>> removeSubsets(List<List<T>> list) {
        List<List<T>> res = new ArrayList<>();
        Set<List<T>> skip = new HashSet<>();

        for (int i = 0; i < list.size(); i++) {
            List<T> item1 = list.get(i);
            if (skip.contains(item1)) {
                continue;
            }
            boolean add = true;
            for (int j = i + 1; j < list.size(); j++) {
                List<T> item2 = list.get(j);
                if (skip.contains(item2)) {
                    continue;
                }
                if (containsInOrder(item1, item2)) {
                    skip.add(item2);
                } else if (containsInOrder(item2, item1)) {
                    skip.add(item1);
                    add = false;
                    break;
                }
            }
            if (add) {
                res.add(item1);
            }
        }
        return res;
    }

    private static List<Map<Integer, String>> merge(List<Map<Integer, String>> listOfMaps) {
        Set<Integer> skipIndexes = new HashSet<>();
        boolean merge = true;
        while (merge) {
            merge = false;
            List<Map<Integer, String>> temp = new ArrayList<>();
            for (int i = 0; i < listOfMaps.size(); i++) {
                if (skipIndexes.contains(i)) {
                    continue;
                }
                boolean mergeCurr = false;
                Map<Integer, String> map1 = listOfMaps.get(i);
                for (int j = i + 1; j < listOfMaps.size(); j++) {
                    if (skipIndexes.contains(j)) {
                        continue;
                    }
                    Map<Integer, String> map2 = listOfMaps.get(j);
                    if (!Collections.disjoint(map1.keySet(), map2.keySet())) {
                        merge = true;
                        mergeCurr = true;
                        Map<Integer, String> newMap = new TreeMap<>();
                        newMap.putAll(map1);
                        newMap.putAll(map2);
                        temp.add(newMap);
                        skipIndexes.add(i);
                        skipIndexes.add(j);
                        break;
                    }
                }
                if (!mergeCurr) {
                    temp.add(map1);
                }
            }
            if (merge) {
                listOfMaps = temp;
            }
        }
        return listOfMaps;
    }

    private static List<List<String>> extract(SemanticGraph sg, Set<String> relations) {
        List<Map<Integer, String>> listOfMaps = new ArrayList<>();
        for (SemanticGraphEdge edge: sg.edgeListSorted()) {
            if (relations.contains(edge.getRelation().getShortName())) {
                boolean added = false;
                IndexedWord governor = edge.getGovernor();
                IndexedWord dependent = edge.getDependent();
                for (Map<Integer, String> map: listOfMaps) {
                    if (map.containsKey(governor.index()) || map.containsKey(dependent.index())) {
                        map.put(governor.index(), governor.originalText());
                        map.put(dependent.index(), dependent.originalText());
                        added = true;
                        break;
                    }
                }
                if (!added) {
                    Map<Integer, String> map = new TreeMap<>();
                    map.put(governor.index(), governor.originalText());
                    map.put(dependent.index(), dependent.originalText());
                    listOfMaps.add(map);
                }
            }
        }
        listOfMaps = merge(listOfMaps);
        return listOfMaps.stream().map(map -> new ArrayList<>(map.values())).collect(Collectors.toList());
    }

    private static List<List<String>> extract(Collection<RelationTriple> triples) {
        List<List<String>> res = new ArrayList<>();
        for (RelationTriple triple: triples) {
            if (triple.confidence == 1.0) {
                List<String> factDescription = new ArrayList<>();
                for (String s: Arrays.asList(triple.subjectGloss(), triple.relationGloss(), triple.objectGloss())) {
                    factDescription.addAll(Arrays.asList(s.trim().split("\\s+")));
                }
                res.add(factDescription);
            }
        }
        return res;
    }

    private static List<List<String>> generate(StanfordCoreNLP pipeline, String text) {
        List<List<String>> res = new ArrayList<>();

        // Annotate an example document.
        Annotation ann = new Annotation(text);
        pipeline.annotate(ann);

        // Relations short names to filter relation triples
        Set<String> relations = new HashSet<>(
                Arrays.asList("nsubj", "nsubjpass", "csubj", "csubjpass", "dobj", "amod", "nummod", "compound")
        );

        // Loop over sentences in the document
        for (CoreMap sentence : ann.get(CoreAnnotations.SentencesAnnotation.class)) {
            // Get SemanticGraph for the sentence
            SemanticGraph sg = sentence.get(SemanticGraphCoreAnnotations.EnhancedDependenciesAnnotation.class);
            res.addAll(extract(sg, relations));

            // Get the OpenIE triples for the sentence
            Collection<RelationTriple> triples = sentence.get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);
            res.addAll(extract(triples));
        }
        return removeSubsets(res);
    }

    public static void main(String[] args) throws IllegalArgumentException, IOException {
        if (args.length != 1) {
            throw new IllegalArgumentException("Requires the file path of paragraphs directory's parent directory");
        }
        String filePath = args[0];
        String inputPath = filePath + "/paragraphs";
        String outputPath = filePath + "/paragraphs_fd";

        // Create the Stanford CoreNLP pipeline
        Properties props = PropertiesUtils.asProperties(
                "annotators", "tokenize,ssplit,pos,lemma,depparse,natlog,openie",
                "depparse.model", DependencyParser.DEFAULT_MODEL
        );
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        File inputDirectory = new File(inputPath);
        File[] inputFiles = inputDirectory.listFiles();
        int numFilesProcessed = 0;
        if (inputFiles != null) {
            for (File inputFile : inputFiles) {
                if (inputFile.isFile()) {
                    // Open file, and get paragraph, summary
                    boolean isSummary = false;
                    StringBuilder paragraphTextBuilder = new StringBuilder();
                    StringBuilder summaryTextBuilder = new StringBuilder();
                    try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                            new FileInputStream(inputFile), StandardCharsets.UTF_8))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            if (line.startsWith("@summary")) {
                                isSummary = true;
                            } else if (isSummary) {
                                summaryTextBuilder.append(line).append("\n");
                            } else {
                                paragraphTextBuilder.append(line).append("\n");
                            }
                        }
                    }
                    String paragraphText = paragraphTextBuilder.toString();
                    String summaryText = summaryTextBuilder.toString();

                    // Pass paragraph text to generate fact descriptions
                    List<List<String>> factDescriptionTokens = generate(pipeline, paragraphText);
                    String factDescriptionText = factDescriptionTokens.stream()
                            .map(lineTokens -> String.join(" ", lineTokens))
                            .map(line -> line + "\n")
                            .collect(Collectors.joining(""));

                    // Write paragraph, fact descriptions and summary to paragraphs_fd directory with same file name
                    String outputFileName = outputPath + "/" + inputFile.getName();
                    File outputFile = new File(outputFileName);
                    outputFile.getParentFile().mkdir();
                    try (OutputStreamWriter writer =
                                 new OutputStreamWriter(new FileOutputStream(outputFile), StandardCharsets.UTF_8)) {
                        writer.write(paragraphText);
                        writer.write("@fact_descriptions\n");
                        writer.write(factDescriptionText);
                        writer.write("@summary\n");
                        writer.write(summaryText);
                    }
                }
                if (++numFilesProcessed % 1000 == 0) {
                    System.out.println(String.format("Number of files processed: %d/%d", numFilesProcessed, inputFiles.length));
                }
            }
        }

    }
}
