<?php

namespace Rubix\ML\Embedders;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Embedders\SoftmaxApproximators\SoftmaxApproximator;
use Rubix\ML\Embedders\SoftmaxApproximators\NegativeSampling;
use Tensor\Matrix;
use Tensor\Vector;
use InvalidArgumentException;
use RuntimeException;

/**
 * Word2Vec
 *
 * A shallow, two-layer neural network, that produces word embeddings used in NLP models.
 * This implementation utilizes the skip-gram algorithm and hierarchical softmax or negative sampling.
 *
 * References:
 * `Tomas Mikolov et al: Efficient Estimation of Word Representations
 * in Vector Space <https://arxiv.org/pdf/1301.3781.pdf>`_, `Tomas Mikolov et al: Distributed Representations of Words
 * and Phrases and their Compositionality <https://arxiv.org/abs/1310.4546>`
 *
 * @category    Machine Learning
 * @package     RubixML
 * @author      Rich Davis
 */
class Word2Vec implements Embedder, Stateful
{
    /**
     * Presetting random multiplier used to validate word probabilities in sub sampling.
     *
     * @var int
     */
    protected const RAND_MULTIPLIER = 4294967296;

    /**
     * The minimum allowed alpha while training.
     *
     * @var float
     */
    protected const MIN_ALPHA = 0.0001;

    /**
     * An array of sanitized and exploded sentences.
     *
     * @var array[]
     */
    protected $corpus = [];

    /**
     * An array of each word in the corpus for preprocessing purposes.
     *
     * @var int[]
     */
    protected $rawVocab = [];

    /**
     * An array containing each word in the corpus and it's respective index, count, and multiplier.
     *
     * @var array[]
     */
    protected $vocab = [];

    /**
     * An array containing each word in the corpus at it's respective index.
     *
     * @var int[]
     */
    protected $index2word = [];

    /**
     * An array of output embeddings.
     *
     * @var Vector[]
     */
    protected $syn1 = [];

    /**
     * An array containing word vectors.
     *
     * @var Vector[]
     */
    protected $vectors = [];

    /**
     * The error vector used during training.
     *
     * @var \Tensor\Vector
     */
    protected $error;

    /**
     * The lock factors for each word in the context.
     *
     * @var int[]
     */
    protected $vectorsLockf = [];

    /**
     * The L2-normalized word vectors from the model.
     *
     * @var Vector[]
     */
    protected $vectorsNorm = [];

    /**
     * The approximation sampling algorithm.
     *
     * @var SoftmaxApproximator
     */
    protected $approximation;

    /**
     * The window size for the skip-gram model.
     *
     * @var int
     */
    protected $window;

    /**
     * The dimensionality of each embedded feature column.
     *
     * @var int
     */
    protected $dimensions;

    /**
     * The degree to which noise words are removed from the training text.
     *
     * @var float
     */
    protected $sampleRate;

    /**
     * The amount of L2 regularization applied to the weights of the output layer.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate
     * over the entire training set before terminating.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The minimum times a word must appear in the corpus to be considered in the training text.
     *
     * @var int
     */
    protected $minCount;

    /**
     * The total number of unique words in the corpus.
     *
     * @var int
     */
    protected $vocabCount;

    /**
     * Sigmoid activation function.
     *
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid
     */
    protected $sigmoid;

    /**
     * @param SoftmaxApproximator|null $approximation
     * @param int $window
     * @param int $dimensions
     * @param float $sampleRate
     * @param float $alpha
     * @param int $epochs
     * @param int $minCount
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $dimensions = 5,
        ?SoftmaxApproximator $approximation = null,
        int $window = 2,
        float $sampleRate = 1e-3,
        float $alpha = 0.01,
        int $epochs = 10,
        int $minCount = 2
    ) {
        if ($dimensions < 5) {
            throw new InvalidArgumentException("Dimensions must be greater than 4, $dimensions given.");
        }

        if ($window > 5) {
            throw new InvalidArgumentException("Window must be between 1 and 5, $window given.");
        }

        if ($sampleRate < 0.0) {
            throw new InvalidArgumentException("Sample rate must be 0 or greater, $sampleRate given.");
        }

        if ($alpha < 0.0) {
            throw new InvalidArgumentException("Alpha must be greater than 0, $alpha given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException("Number of epochs must be greater than 0, $epochs given.");
        }

        if ($minCount < 1) {
            throw new InvalidArgumentException("Minimum word count must be greater than 0, $minCount given.");
        }

        $this->approximation = $approximation ?? new NegativeSampling();
        $this->dimensions = $dimensions;
        $this->window = $window;        
        $this->sampleRate = $sampleRate;
        $this->alpha = $alpha;
        $this->epochs = $epochs;
        $this->minCount = $minCount;
        $this->sigmoid = new Sigmoid();
    }

    /**
     * Return the vocab array.
     *
     * @return mixed[]
     */
    public function vocab() : array
    {
        return $this->vocab;
    }

    /**
     * Return the vocab count.
     *
     * @return int
     */
    public function vocabCount() : int
    {
        return $this->vocabCount;
    }

    /**
     * Return the index2word array.
     *
     * @return int[]
     */
    public function index2word() : array
    {
        return $this->index2word;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::categorical(),
        ];
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'layer' => $this->approximation,
            'dimensions' => $this->dimensions,
            'window' => $this->window,            
            'sample_rate' => $this->sampleRate,
            'alpha' => $this->alpha,
            'epochs' => $this->epochs,
            'min_count' => $this->minCount,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return !empty($this->vectors);
    }

    /**
     * Iterate through a range of the specific epoch count and updating all respective word vectors.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithTransformer::check($dataset, $this);

        if($dataset->numColumns() > 1){
            throw new InvalidArgumentException("Only datasets with 1 column are supported.");
        }

        $sentences = $dataset->column(0);

        $this->preprocess($sentences);
        $this->prepareWeights();

        $startAlpha = $this->alpha;

        for ($i = 0; $i < $this->epochs; ++$i) {
            $this->alpha = $startAlpha - (($startAlpha - self::MIN_ALPHA) * ($i) / $this->epochs);

            $this->trainEpochSg();
        }

        $this->generateL2Norm();
        unset($this->error);
    }

    /**
     * Return the word embedding for a given word.
     *
     * @param string $word
     * @param bool $useNorm
     * @return \Tensor\Vector|null $result
     */
    public function wordVec(string $word, bool $useNorm = true) : ?Vector
    {
        if (!isset($this->vocab[$word])) {
            return null;
        }

        if ($useNorm) {
            return $this->vectorsNorm[$this->vocab[$word]['index']];
        }

        return $this->vectors[$this->vocab[$word]['index']];
    }

    /**
     * Return the word embedding, or a vector of zeros if empty, for a given word.
     *
     * @param string $word
     * @param bool $useNorm
     * @return \Tensor\Vector $result
     */
    public function embedWord(string $word, bool $useNorm = true) : Vector
    {
        $wordEmbedding = $this->wordVec($word, $useNorm);

        if (!$wordEmbedding) {
            $wordEmbedding = Vector::zeros($this->dimensions);
        }

        return $wordEmbedding;
    }

    /**
     * Embed a high dimensional dataset into a lower dimensional one.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (!$this->vectorsNorm) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        SamplesAreCompatibleWithTransformer::check(new Unlabeled($samples), $this);

        foreach ($samples as &$sample) {
            foreach ($sample as $i2 => $sentence) {
                $preppedSentence = $this->prepSentence($sentence);

                $embeddings = [];
                foreach ($preppedSentence as $word) {
                    $embeddings[] = $this->embedWord($word);
                }
            }

            if (isset($embeddings)) {
                $sample = Matrix::stack($embeddings)->transpose()->mean()->asArray();
            }
        }
    }

    /**
     * Determine the top N similar words provided an array of positive words and negative words.
     *
     * @param string[] $positive
     * @param string[] $negative
     * @param int|null $top
     * @return string[] $result
     */
    public function mostSimilar(array $positive, array $negative = [], ?int $top = 20) : array
    {
        $positiveArray = $negativeArray = $allWords = $means = [];

        foreach ($positive as $word) {
            $positiveArray[$word] = 1.0;
        }

        foreach ($negative as $word) {
            $negativeArray[$word] = -1.0;
        }

        $wordArray = array_merge($positiveArray, $negativeArray);

        foreach ($wordArray as $word => $weight) {
            $wordEmbedding = $this->wordVec($word);

            if ($wordEmbedding instanceof Vector) {
                $means[] = $wordEmbedding->multiplyScalar($weight);
                $allWords[] = $this->vocab[$word]['index'];
            }
        }

        if (empty($allWords)) {
            throw new InvalidArgumentException('Positive words were not found in vocab.');
        }

        $mean = Matrix::stack($means)->transpose()->mean();
        $l2 = Matrix::stack($this->vectorsNorm);
        $dists = $mean->transpose()->matmul($l2->transpose())->asArray()[0];

        arsort($dists);

        $result = [];
        foreach ($dists as $index => $weight) {
            if (!in_array($index, $allWords)) {
                $result[$this->index2word[$index]] = $weight;
            }
        }

        return array_slice($result, 0, $top, true);
    }

    /**
     * Update vector weights and return error.
     *
     * @param mixed[] $predictWord
     * @param \Tensor\Vector $l1
     * @return \Tensor\Vector
     */
    protected function error($predictWord, $l1) : Vector
    {
        $wordIndices = $this->approximation->wordIndices($predictWord);
        $l2 = $this->layerMatrix($wordIndices);
        $fa = $this->propagateHidden($l2, $l1);
        $gradient = $this->approximation->gradientDescent($fa, $predictWord['word'], $this->alpha);

        $this->learnHidden($wordIndices, $gradient, $l1);

        return $this->error->addMatrix($gradient->matmul($l2))->rowAsVector(0);
    }

    /**
     * Scan vocab, prepare vocab, sort vocab, set sampling methods
     *
     * @param string[] $sentences
     */
    private function preprocess(array $sentences) : void
    {
        $words = $this->prepCorpus($sentences);

        $this->scanVocab($words);
        $this->prepVocab();
        $this->sortVocab();

        $this->approximation->structureSampling($this);
    }

    /**
     * Parse, sanitize, & prepare an array of sentences to generate & set corpus.
     *
     * @param string[] $sentences
     * @return string[] $words
     */
    private function prepCorpus(array $sentences) : array
    {
        $words = [];

        foreach ($sentences as $sentence) {
            $preppedSentence = $this->prepSentence($sentence);

            $this->corpus[] = $preppedSentence;
            $words = array_merge($words, $preppedSentence);
        }

        return $words;
    }

    /**
     * Sanitize & explode a provided sentence.
     *
     * @param string $sentence
     * @return string[] $exploded
     */
    private function prepSentence(string $sentence) : array
    {
        $sentence = (string) preg_replace('#[[:punct:]]#', '', strtolower($sentence));
        $sentence = (string) preg_replace('/\s\s+/', ' ', str_replace("\n", ' ', $sentence));
        $sentence = trim($sentence);

        return explode(' ', $sentence);
    }

    /**
     * Prepare and set an initial raw vocab array for additional preprocessing
     *
     * @param string[] $words
     */
    private function scanVocab(array $words) : void
    {
        foreach ($words as $word) {
            $this->rawVocab[$word] = (int) ($this->rawVocab[$word] ?? 0) + 1;
        }
    }

    /**
     * Iterate through each word in raw vocab and formally appending word, and corresponding word count, index, and word, to vocab property.
     * Create initial index2word property to manage word counts.
     */
    private function prepVocab() : void
    {
        $dropTotal = $dropUnique = $retainTotal = 0;
        $retainWords = [];

        foreach ($this->rawVocab as $word => $v) {
            if ($v >= $this->minCount) {
                $retainWords[] = $word;
                $retainTotal += $v;

                $this->vocab[$word] = ['count' => $v, 'index' => count($this->index2word), 'word' => $word];
                $this->index2word[] = $word;
            } else {
                ++$dropUnique;
                $dropTotal += $v;
            }
        }

        $originalUniqueTotal = count($retainWords) + $dropUnique;
        $retainUniquePct = (count($retainWords) * 100) / $originalUniqueTotal;

        $originalTotal = $retainTotal + $dropTotal;
        $retainPct = ($retainTotal * 100) / $originalTotal;

        $thresholdCount = $this->thresholdCount($retainTotal);

        $this->updateWordProbs($retainWords, $thresholdCount);
    }

    /**
     * Determine threshold word count based off of subsampling rate.
     *
     * @param int $retainTotal
     * @return float
     */
    private function thresholdCount(int $retainTotal) : float
    {
        if (!$this->sampleRate) {
            return $retainTotal;
        }
        if ($this->sampleRate < 1) {
            return $this->sampleRate * $retainTotal;
        }

        return $this->sampleRate * (3 + sqrt(5)) / 2;
    }

    /**
     * Assign word probabilities, determined by subsampling rate, to each word in vocabulary.
     *
     * @param string[] $retainWords
     * @param float $thresholdCount
     */
    private function updateWordProbs(array $retainWords, float $thresholdCount) : void
    {
        $downsampleTotal = $downsampleUnique = 0;

        foreach ($retainWords as $w) {
            $v = $this->rawVocab[$w];
            $wordProbability = (sqrt($v / $thresholdCount) + 1) * ($thresholdCount / $v);

            if ($wordProbability < 1) {
                ++$downsampleUnique;
                $downsampleTotal += $wordProbability * $v;
            } else {
                $wordProbability = 1;
                $downsampleTotal += $v;
            }

            $this->vocab[$w]['sample_int'] = round($wordProbability * 2 ** 32);
        }
    }

    /**
     * Sort vocabulary, create index2word property, and assign respective word index to each vocabulary word.
     */
    private function sortVocab() : void
    {
        $original = $this->vocab;

        $count = array_column($original, 'count');
        array_multisort($count, SORT_DESC, $original);

        $this->index2word = array_column($original, 'word');

        foreach ($this->index2word as $index => $word) {
            $this->vocab[$word]['index'] = $index;
        }

        $this->vocabCount = count($this->vocab);
    }

    /**
     * Assign random vector for each word in the corpus, instead of creating a massive random matrix for RAM purposes.
     * Create zeroed vectors for syn and error.
     */
    private function prepareWeights() : void
    {
        $zeroVector = Vector::zeros($this->dimensions);

        for ($i = 0; $i < $this->vocabCount; ++$i) {
            $this->syn1[] = $zeroVector;
            $this->vectors[$i] = Vector::rand($this->dimensions)->subtractScalar(0.5)->divideScalar($this->dimensions);
        }

        $this->error = $zeroVector;
        $this->vectorsLockf = array_fill(0, $this->vocabCount, 1);
    }

    /**
     * Train one epoch from the corpus and updating all respective word vectors.
     */
    private function trainEpochSg() : void
    {
        foreach ($this->corpus as $sentence) {
            $wordVocabs = $this->wordVocabs($sentence);

            foreach ($wordVocabs as $pos => $word) {
                $subset = $this->sgSubset($pos, $wordVocabs);

                foreach ($subset as $pos2 => $word2) {
                    if ($pos2 !== $pos) {
                        $wordIndex = (string) $this->index2word[$word['index']];
                        $contextIndex = $word2['index'];

                        $this->trainPairSg($wordIndex, $contextIndex);
                    }
                }
            }
        }
    }

    /**
     * Build an array of Word Vocabs that exceed a random multiplier from the sentence for more accurate and faster training
     *
     * @param string[] $sentence
     * @return array[] $wordVocabs
     */
    private function wordVocabs(array $sentence) : array
    {
        $wordVocabs = [];
        $rand = rand() / getrandmax();
        $randMultiplier = $rand * self::RAND_MULTIPLIER;

        foreach ($sentence as $word) {
            $vocabItem = $this->vocab[$word] ?? false;

            if (!empty($vocabItem) && $vocabItem['sample_int'] > $randMultiplier) {
                $wordVocabs[] = $vocabItem;
            }
        }

        return $wordVocabs;
    }

    /**
     * Build an array from the word vocab in skip-gram sequence
     *
     * @param int $pos
     * @param array[] $wordVocabs
     * @return array[]
     */
    private function sgSubset(int $pos, array $wordVocabs) : array
    {
        $reducedWindow = rand(0, ($this->window - 1));
        $arrayStart = max(0, ($pos - $this->window + $reducedWindow));
        $arrayEnd = $pos + $this->window + 1 - $reducedWindow - $arrayStart;

        return array_slice($wordVocabs, $arrayStart, $arrayEnd, true);
    }

    /**
     * Determine appropriate word pair training method and updating the word's vector weights.
     *
     * @param string $wordIndex
     * @param int $contextIndex
     */
    private function trainPairSg(string $wordIndex, int $contextIndex) : void
    {
        $predictWord = $this->vocab[$wordIndex];
        $l1 = $this->vectors[$contextIndex];
        $lockFactor = $this->vectorsLockf[$contextIndex];
        $error = $this->error($predictWord, $l1);

        $this->vectors[$contextIndex] = $l1->addVector($error->multiplyScalar($lockFactor));
    }

    /**
     * Create 2-d matrix from word indices.
     *
     * @param mixed[] $wordIndices
     * @return \Tensor\Matrix
     */
    private function layerMatrix(array $wordIndices) : Matrix
    {
        $l2a = [];

        foreach ($wordIndices as $index) {
            $l2a[] = $this->syn1[$index];
        }

        return Matrix::stack($l2a);
    }

    /**
     * Update the hidden layer for given word index supplied the outer product of the word vector and error gradients.
     *
     * @param mixed[] $wordIndices
     * @param \Tensor\Vector $g
     * @param \Tensor\Vector $l1
     */
    private function learnHidden(array $wordIndices, Vector $g, Vector $l1) : void
    {
        $c = $g->outer($l1);

        foreach ($wordIndices as $i=>$index) {
            $this->syn1[$index] = $this->syn1[$index]->addVector($c->rowAsVector($i));            
        }
    }

    /**
     * Propagate hidden layer.
     *
     * @param \Tensor\Matrix $l2
     * @param \Tensor\Vector $l1
     * @return \Tensor\Vector
     */
    private function propagateHidden(Matrix $l2, Vector $l1) : Vector
    {
        $prodTerm = $l1->matmul($l2->transpose());

        return $this->sigmoid->compute($prodTerm)->rowAsVector(0);
    }

    /**
     * Generate L2 Norm of final word vectors.
     */
    private function generateL2Norm() : void
    {
        $l2Norm = [];

        foreach ($this->vectors as $vector) {
            $l2Norm[] = $vector->divideScalar($vector->L2Norm());
        }

        $this->vectorsNorm = $l2Norm;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Word2Vec {' . Params::stringify($this->params()) . '}';
    }
}
