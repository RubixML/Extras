<?php

namespace Rubix\ML\Tests\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Persisters\Serializers\Bzip2;
use Rubix\ML\Persisters\Serializers\Serializer;
use PHPUnit\Framework\TestCase;

/**
 * @group Serializers
 * @requires extension bz2
 * @covers \Rubix\ML\Persisters\Serializers\Bzip2
 */
class Bzip2Test extends TestCase
{
    /**
     * @var \Rubix\ML\Persistable
     */
    protected $persistable;

    /**
     * @var \Rubix\ML\Persisters\Serializers\Bzip2
     */
    protected $serializer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->serializer = new Bzip2(4, 0);

        $this->persistable = new DummyClassifier();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Bzip2::class, $this->serializer);
        $this->assertInstanceOf(Serializer::class, $this->serializer);
    }

    /**
     * @test
     */
    public function serializeUnserialize() : void
    {
        $data = $this->serializer->serialize($this->persistable);

        $this->assertInstanceOf(Encoding::class, $data);

        $persistable = $this->serializer->unserialize($data);

        $this->assertInstanceOf(DummyClassifier::class, $persistable);
        $this->assertInstanceOf(Persistable::class, $persistable);
    }
}
