// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import sfa.classification.Classifier.Words;
import sfa.timeseries.TimeSeries;

import com.carrotsearch.hppc.ObjectIntOpenHashMap;
import com.carrotsearch.hppc.cursors.IntCursor;

/**
 * Symbolic Fourier Approximation as published in
 *    Schäfer, P., Högqvist, M.: SFA: a symbolic fourier approximation and 
 *    index for similarity search in high dimensional datasets. 
 *    In: EDBT, ACM (2012)
 */
public class SFA implements Serializable {
  private static final long serialVersionUID = -3903361341617350743L;

  // distribution of Fourier values
  public ArrayList<ValueLabel>[] orderLine;

  public HistogramType histogramType = HistogramType.EQUI_DEPTH;

  public int alphabetSize = 256;
  public byte neededBits = (byte)Words.binlog(this.alphabetSize);
  public int wordLength = 0;
  public boolean initialized = false;
  public boolean lowerBounding = true;
  
  public int maxWordLength;

  // The Momentary Fourier Transform
  public MFT transformation;
  
  // use binning / bucketing
  public double[][] bins;

  public enum HistogramType {
    EQUI_FREQUENCY, EQUI_DEPTH, INFORMATION_GAIN
  }

  static class ValueLabel implements Serializable {
    private static final long serialVersionUID = 4392333771929261697L;

    public double value;
    public String label;
    public ValueLabel(double key, String  label) {
      this.value = key;
      this.label = label;
    }
    @Override
    public String toString() {
      return "" + this.value + ":" + this.label;
    }
  }

  public SFA(HistogramType historgramType) {
    reset();
    this.histogramType = historgramType;
  }
 
  public void reset() {
    this.initialized = false;
    this.orderLine = null;
    this.bins = null;
  }

  @SuppressWarnings("unchecked")
  public void init(int l, int alphabetSize) {
    this.wordLength = l;
    this.maxWordLength = l;
    this.alphabetSize = alphabetSize;
    this.initialized = true;

    // l-dimensional bins    
    this.alphabetSize = alphabetSize;
    this.neededBits = (byte)Words.binlog(alphabetSize);

    this.bins = new double[l][alphabetSize-1];
    for (double[] row : this.bins ) {
      Arrays.fill(row, Double.MAX_VALUE);
    }

    this.orderLine = new ArrayList[l];
    for (int i = 0; i < this.orderLine.length; i++) {
      this.orderLine[i] = new ArrayList<ValueLabel>();
    }
  }

  /**
   * Transforms a single time series to its SFA word
   * @param timeSeries
   * @return
   */
  public short[] transform(TimeSeries timeSeries) {
    return transform(timeSeries, null);
  }

  /**
   * Transforms a single time series to its SFA word
   * @param timeSeries
   * @param approximation the DFT approximation, if available, else pass 'null'
   * @return
   */
  public short[] transform(TimeSeries timeSeries, double[] approximation) {
    if (!initialized) {
      throw new RuntimeException("Plase call fitTransform() first.");
    }
    if (approximation == null) {
      // get approximation of the time series
      approximation = this.transformation.transform(timeSeries, this.maxWordLength);
    }

    // use lookup table (bins) to get the word from the approximation
    return quantization(approximation);
  }

  /**
   * Transforms a set of samples to SFA words
   * @param samples
   * @return
   */
  public short[][] transform(TimeSeries[] samples) {
    if (!initialized) {
      throw new RuntimeException("Plase call fitTransform() first.");
    }
    short[][] transform = new short[samples.length][];
    for (int i = 0; i < transform.length; i++) {
      transform[i] = transform(samples[i], null);
    }

    return transform;
  }
  
  /**
   * Transforms a set of time series to SFA words.
   * @param samples
   * @param approximation the DFT approximations, if available, else pass 'null'
   * @return
   */
  public short[][] transform(TimeSeries[] samples, double[][] approximation) {
    if (!initialized) {
      throw new RuntimeException("Plase call fitTransform() first.");
    }    
    short[][] transform = new short[samples.length][];
    for (int i = 0; i < transform.length; i++) {
      transform[i] = transform(samples[i], approximation[i]);
    }

    return transform;
  }

  /**
   * Quantization of a DFT approximation to its SFA word
   * @param approximation the DFT approximation of a time series
   * @return
   */
  public short[] quantization(double[] approximation) {
    int i = 0;
    short[] word = new short[approximation.length];
    for (double value : approximation) {
      // lookup character:
      short c = 0;
      for (c = 0; c < this.bins[i].length; c++) {
        if (value < this.bins[i][c]) {
          break;
        }
      }
      word[i++] = c;
    }
    return word;
  }
  
  public byte[] quantizationByte(double[] approximation) {
    int i = 0;
    byte[] word = new byte[approximation.length];
    for (double value : approximation) {
      // lookup character:
      byte c = 0;
      for (c = 0; c < this.bins[i].length; c++) {
        if (value < this.bins[i][c]) {
          break;
        }
      }
      word[i++] = c;
    }
    return word;
  }

  protected void sortOrderLine() {
    for (List<ValueLabel> element : this.orderLine) {
      Collections.sort(element, new Comparator<ValueLabel>() {
        @Override
        public int compare(ValueLabel o1, ValueLabel o2) {
          int comp = Double.compare(o1.value, o2.value);
          if (comp != 0) {
            return comp;
          }
          return o1.label != null ? o1.label.compareTo(o2.label) : 0;
        }
      });
    }
  }

  /**
   * Extracts sliding windows from the time series and trains SFA based on the sliding windows. 
   * At the end of this call, the quantization bins are set.
   * @param timeSeries
   * @param windowLength The length of each sliding window
   * @param wordLength the SFA word-length
   * @param symbols the SFA alphabet size
   * @param normMean if set, the mean is subtracted from each sliding window
   */
  public void fitWindowing(TimeSeries[] timeSeries, int windowLength, int wordLength, int symbols, boolean normMean, boolean lowerBounding) {
    this.transformation = new MFT(windowLength, normMean, lowerBounding);

    ArrayList<TimeSeries> sa = new ArrayList<TimeSeries>(timeSeries.length * timeSeries[0].getLength() / windowLength);
    for (TimeSeries t : timeSeries) {
      sa.addAll(Arrays.asList(t.getDisjointSequences(windowLength, normMean)));
    }
    fitTransform(sa.toArray(new TimeSeries[]{}), wordLength, symbols, normMean);
  }

  /**
   * Extracts sliding windows from a time series and transforms it to its SFA word.
   * @param ts
   * @param windowLength
   * @param wordLength
   * @return
   */
  public short[][] transformWindowing(TimeSeries ts, int wordLength) {
    double[][] mft = this.transformation.transformWindowing(ts, maxWordLength);
    
    short[][] words = new short[mft.length][];
    for (int i = 0; i < mft.length; i++) {
      words[i] = quantization(mft[i]);      
    }
    
    return words;
  }
  
  /**
   * Extracts sliding windows from a time series and applies the Fourier Transform.
   * @param ts
   * @param windowLength
   * @param wordLength
   * @return
   */
  public double[][] transformWindowingDouble(TimeSeries ts, int wordLength) {
    return this.transformation.transformWindowing(ts, maxWordLength);    
  }

//  public float[][] transformWindowingFloat(TimeSeries ts, int wordLength) {
//    double[][] doubleData = this.transformation.transformWindowing(ts, maxWordLength);
//    float[][] floatData = new float[doubleData.length][doubleData[0].length];
//    for (int i = 0; i < doubleData.length; i++) {
//      for (int j = 0; j < doubleData[i].length; j++) {
//        floatData[i][j] =  (float) doubleData[i][j];
//      }
//    }
//    return floatData;
//  }

  /**
   * Extracts sliding windows from a time series and transforms it to its SFA word.
   * @param ts
   * @param windowLength
   * @param wordLength
   * @return
   */
  public int[] transformWindowingInt(TimeSeries ts, int wordLength) {
    short[][] words = transformWindowing(ts, wordLength);
    int[] intWords = new int[words .length];
    for (int i = 0; i < words .length; i++) {
      intWords[i] = (int)Words.createWord(words[i], wordLength, neededBits);
    }
    return intWords;
  }

  public double[][] fitTransformDouble (TimeSeries[] samples, int wordLength, int symbols, boolean normMean) {
    if (!this.initialized) {
      init(wordLength, symbols);
      
      if (this.transformation == null) { // TODO!!
        this.transformation = new MFT(samples[0].getLength(), normMean, lowerBounding);
      }
    }

    double[][] transformedSamples = fillOrderline(samples, wordLength, symbols);

    if (this.histogramType == HistogramType.EQUI_DEPTH) {
      divideEquiDepthHistogram();
    } else if (this.histogramType == HistogramType.EQUI_FREQUENCY) {
      divideEquiWidthHistogram();
    } else if (this.histogramType == HistogramType.INFORMATION_GAIN) {
      divideHistogramInformationGain();
    }
    
    return transformedSamples;
  }

  /**
   * Trains SFA based on a set of samples. 
   * At the end of this call, the quantization bins are set.
   * @param samples
   * @param wordLength
   * @param symbols
   * @return
   */
  public short[][] fitTransform (TimeSeries[] samples, int wordLength, int symbols, boolean normMean) {      
    return transform(samples, fitTransformDouble(samples, wordLength, symbols, normMean));
  }
  
  /**
   * Fill data in the orderline
   * @param samples
   */
  protected double[][] fillOrderline (TimeSeries[] samples, int l, int symbols) {
    double[][] transformedSamples = new double[samples.length][];

    for (int i = 0; i < samples.length; i++) {
      // z-normalization
      samples[i].norm();

      // approximation
      transformedSamples[i] = this.transformation.transform(samples[i], l);

      for (int j=0; j < transformedSamples[i].length; j++) {
        // round to 2 decimal places to reduce noise
        double value = Math.round(transformedSamples[i][j]*100.0)/100.0;
        this.orderLine[j].add(new ValueLabel(value, samples[i].getLabel()));
      }
    }

    // Sort ascending by value
    sortOrderLine();
    
    return transformedSamples;
  }

  /**
   * Use equi-width binning to divide the orderline
   */
  protected void divideEquiWidthHistogram () {
    int i = 0;
    for (List<ValueLabel> elements : this.orderLine) {
      if (!elements.isEmpty()) {
        // apply the split
        double first = elements.get(0).value;
        double last = elements.get(elements.size()-1).value;
        double intervalWidth = (last-first) / (double)(this.alphabetSize);

        for (int c = 0; c < this.alphabetSize-1; c++) {
          this.bins[i][c] = intervalWidth*(c+1)+first;
        }
      }
      i++;
    }
  }

  /**
   * Use equi-depth binning to divide the orderline
   */
  protected void divideEquiDepthHistogram () {
    // For each real and imaginary part
    for (int i = 0; i < this.bins.length; i++) {
      // Divide into equi-depth intevals
      double depth = this.orderLine[i].size() / (double)(this.alphabetSize);

      int pos = 0;
      long count = 0;
      for (ValueLabel value : this.orderLine[i]) {
        if (++count > Math.ceil(depth*(pos+1))
            && (pos==0 || this.bins[i][pos-1] != value.value)) {
          this.bins[i][pos++] = value.value;
        }
      }
    }
  }

  /**
   * Use information-gain to divide the orderline
   */
  protected void divideHistogramInformationGain() {
    // for each Fourier coefficient: split using maximal information gain
    for (int i = 0; i < this.orderLine.length; i++) {
      List<ValueLabel> element = this.orderLine[i];
      if (!element.isEmpty()) {
        ArrayList<Integer> splitPoints = new ArrayList<Integer>();
        findBestSplit(element, 0, element.size(), this.alphabetSize, splitPoints);

        Collections.sort(splitPoints);

        // apply the split
        for (int j = 0; j < splitPoints.size(); j++) {
          double value = element.get(splitPoints.get(j)+1).value;
//          double value = (element.get(splitPoints.get(j)).value + element.get(splitPoints.get(j)+1).value)/2.0;
          this.bins[i][j] = value;
        }
      }
    }
  }
  
  protected static double entropy(ObjectIntOpenHashMap<String> frequency, double total) {
    double entropy = 0;
    double log2 = 1.0 / Math.log(2.0);
    for (IntCursor element : frequency.values()) {
      double p = element.value / total;
      if (p > 0) {
        entropy -= p * Math.log(p) * log2;
      }
    }
    return entropy;
  }

  protected static double calculateInformationGain(
      ObjectIntOpenHashMap<String> cIn,
      ObjectIntOpenHashMap<String> cOut,
      double class_entropy,
      double total_c_in,
      double total) {
    double total_c_out = (total - total_c_in);
    return class_entropy
           - total_c_in / total * entropy(cIn, total_c_in)
           - total_c_out / total * entropy(cOut, total_c_out);
  }

  protected void findBestSplit(
      List<ValueLabel> element,
      int start,
      int end,
      int remainingSymbols,
      List<Integer> splitPoints
      ) {

    double bestGain = -1;
    int bestPos = -1;

    // class entropy
    int total = end-start;
    ObjectIntOpenHashMap<String> cIn = new ObjectIntOpenHashMap<String>();
    ObjectIntOpenHashMap<String> cOut = new ObjectIntOpenHashMap<String>();
    for (int pos = start; pos < end; pos++) {
      cOut.putOrAdd(element.get(pos).label, 1, 1);
    }
    double class_entropy = entropy(cOut, total);

    int i = start;
    String lastLabel = element.get(i).label;
    i += moveElement(element, cIn, cOut, start);

    for (int split = start+1; split < end-1; split++) {
      String label = element.get(i).label;
      i += moveElement(element, cIn, cOut, split);

      // only inspect changes of the label
      if (!label.equals(lastLabel)) {
        double gain = calculateInformationGain(cIn, cOut, class_entropy, i, total);

        if (gain >= bestGain) {
          bestPos = split;
          bestGain = gain;
        }
      }
      lastLabel = label;
    }

    if (bestPos > -1) {
      splitPoints.add(bestPos);

      // recursive split
      remainingSymbols = remainingSymbols / 2;
      if (remainingSymbols > 1) {
        if (bestPos - start > 2 && end - bestPos > 2) { // enough data points?
          findBestSplit(element, start, bestPos, remainingSymbols, splitPoints);
          findBestSplit(element, bestPos, end, remainingSymbols, splitPoints);
        }
        else if (end - bestPos > 4) { // enough data points?
          findBestSplit(element, bestPos, (end-bestPos)/2, remainingSymbols, splitPoints);
          findBestSplit(element, (end-bestPos)/2, end, remainingSymbols, splitPoints);
        }
        else if (bestPos - start > 4) { // enough data points?
          findBestSplit(element, start, (bestPos-start)/2, remainingSymbols, splitPoints);
          findBestSplit(element, (bestPos-start)/2, end, remainingSymbols, splitPoints);
        }
      }
    }
  }

  protected int moveElement(
      List<ValueLabel> element,
      ObjectIntOpenHashMap<String> cIn, ObjectIntOpenHashMap<String> cOut,
      int pos) {
    cIn.putOrAdd(element.get(pos).label, 1, 1);
    cOut.putOrAdd(element.get(pos).label, -1, -1);
    return 1;
  }
  
  public void printBins () {
    System.out.print("[");
    for (double[] element : this.bins) {
      System.out.print("-Inf\t" );
      for (double element2 : element) {
        String e = element2 != Double.MAX_VALUE? (""+element2) : "Inf";
        System.out.print("," + e + "\t");
      }
      System.out.println(";");
    }
    System.out.println("]");
  }
  
  public static SFA loadFromDisk(String path) {
    try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path));) {
      return (SFA) in.readObject();
    }
    catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  public boolean writeToDisk(String path) {
    try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path))) {
      out.writeObject(this);
      return true;
    } catch (IOException e) {
      e.printStackTrace();
    }
    return false;
  }
}
