<?php

namespace Rubix\ML\Tests\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Serializers\Igbinary;
use Rubix\ML\Serializers\Serializer;
use Rubix\ML\Classifiers\DummyClassifier;
use PHPUnit\Framework\TestCase;

/**
 * @group Serializers
 * @requires extension igbinary
 * @covers \Rubix\ML\Persisters\Serializers\Igbinary
 */
class IgbinaryTest extends TestCase
{
    /**
     * @var \Rubix\ML\Persistable
     */
    protected $persistable;

    /**
     * @var \Rubix\ML\Serializers\Igbinary
     */
    protected $serializer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->serializer = new Igbinary();

        $this->persistable = new DummyClassifier();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Igbinary::class, $this->serializer);
        $this->assertInstanceOf(Serializer::class, $this->serializer);
    }

    /**
     * @test
     */
    public function serializeDeserialize() : void
    {
        $data = $this->serializer->serialize($this->persistable);

        $this->assertInstanceOf(Encoding::class, $data);

        $persistable = $this->serializer->deserialize($data);

        $this->assertInstanceOf(DummyClassifier::class, $persistable);
        $this->assertInstanceOf(Persistable::class, $persistable);
    }
}
