package com.bosuyun.ai.serving;

/**
 * Created by liuyuancheng on 2021/9/2  <br/>
 *
 * @author liuyuancheng
 */
/*
 * Copyright (c) 2019, 2021, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */

import ai.onnxruntime.*;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

/** Tests for the onnx-runtime Java interface. */
public class InferenceTest {
    private static final Logger logger = Logger.getLogger(InferenceTest.class.getName());
    private static final Pattern LOAD_PATTERN = Pattern.compile("[,\\[\\] ]");

    private static final String propertiesFile = "Properties.txt";

    private static final Pattern inputPBPattern = Pattern.compile("input_*.pb");
    private static final Pattern outputPBPattern = Pattern.compile("output_*.pb");

    private static Path getResourcePath(String path) {
        return new File(InferenceTest.class.getResource(path).getFile()).toPath();
    }

    @Test
    public void repeatedCloseTest() throws OrtException {
        Logger.getLogger(OrtEnvironment.class.getName()).setLevel(Level.SEVERE);
        OrtEnvironment env = OrtEnvironment.getEnvironment("repeatedCloseTest");
        try (OrtEnvironment otherEnv = OrtEnvironment.getEnvironment()) {
            assertFalse(otherEnv.isClosed());
        }
        assertFalse(env.isClosed());
        env.close();
        assertTrue(env.isClosed());
    }

    @Test
    public void morePartialInputsTest() throws OrtException {
        String modelPath = getResourcePath("/partial-inputs-test-2.onnx").toString();
        try (OrtEnvironment env =
                     OrtEnvironment.getEnvironment(
                             OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL, "partialInputs");
             OrtSession.SessionOptions options = new SessionOptions();
             OrtSession session = env.createSession(modelPath, options)) {
            assertNotNull(session);
            assertEquals(3, session.getNumInputs());
            assertEquals(1, session.getNumOutputs());

            // Input and output collections.
            Map<String, OnnxTensor> inputMap = new HashMap<>();
            Set<String> requestedOutputs = new HashSet<>();

            BiFunction<Result, String, Float> unwrapFunc =
                    (r, s) -> {
                        try {
                            return ((float[]) r.get(s).get().getValue())[0];
                        } catch (OrtException e) {
                            return Float.NaN;
                        }
                    };

            // Graph has three scalar inputs, a, b, c, and a single output, ab.
            OnnxTensor a = OnnxTensor.createTensor(env, new float[] {2.0f});
            OnnxTensor b = OnnxTensor.createTensor(env, new float[] {3.0f});
            OnnxTensor c = OnnxTensor.createTensor(env, new float[] {5.0f});

            // Request all outputs, supply all inputs
            inputMap.put("a:0", a);
            inputMap.put("b:0", b);
            inputMap.put("c:0", c);
            requestedOutputs.add("ab:0");
            try (Result r = session.run(inputMap, requestedOutputs)) {
                assertEquals(1, r.size());
                float abVal = unwrapFunc.apply(r, "ab:0");
                assertEquals(6.0f, abVal, 1e-10);
            }

            // Don't specify an output, expect all of them returned.
            try (Result r = session.run(inputMap)) {
                assertEquals(1, r.size());
                float abVal = unwrapFunc.apply(r, "ab:0");
                assertEquals(6.0f, abVal, 1e-10);
            }

            inputMap.clear();
            requestedOutputs.clear();

            // Request single output ab, supply required inputs
            inputMap.put("a:0", a);
            inputMap.put("b:0", b);
            requestedOutputs.add("ab:0");
            try (Result r = session.run(inputMap, requestedOutputs)) {
                assertEquals(1, r.size());
                float abVal = unwrapFunc.apply(r, "ab:0");
                assertEquals(6.0f, abVal, 1e-10);
            }
            inputMap.clear();
            requestedOutputs.clear();

            // Request output but don't supply the inputs
            inputMap.put("c:0", c);
            requestedOutputs.add("ab:0");
            try (Result r = session.run(inputMap, requestedOutputs)) {
                fail("Expected to throw OrtException due to incorrect inputs");
            } catch (OrtException e) {
                // System.out.println(e.getMessage());
                // pass
            }
            inputMap.clear();
            requestedOutputs.clear();

            // Request output but don't supply all the inputs
            inputMap.put("b:0", b);
            requestedOutputs.add("ab:0");
            try (Result r = session.run(inputMap, requestedOutputs)) {
                fail("Expected to throw OrtException due to incorrect inputs");
            } catch (OrtException e) {
                // System.out.println(e.getMessage());
                // pass
            }
        }
    }

    @Test
    public void partialInputsTest() throws OrtException {
        String modelPath = getResourcePath("/partial-inputs-test.onnx").toString();
        try (OrtEnvironment env =
                     OrtEnvironment.getEnvironment(
                             OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL, "partialInputs");
             OrtSession.SessionOptions options = new SessionOptions();
             OrtSession session = env.createSession(modelPath, options)) {
            assertNotNull(session);
            assertEquals(3, session.getNumInputs());
            assertEquals(3, session.getNumOutputs());

            // Input and output collections.
            Map<String, OnnxTensor> inputMap = new HashMap<>();
            Set<String> requestedOutputs = new HashSet<>();

            BiFunction<Result, String, Float> unwrapFunc =
                    (r, s) -> {
                        try {
                            return ((float[]) r.get(s).get().getValue())[0];
                        } catch (OrtException e) {
                            return Float.NaN;
                        }
                    };

            // Graph has three scalar inputs, a, b, c, and three outputs, ab, bc, ab + bc.
            OnnxTensor a = OnnxTensor.createTensor(env, new float[] {2.0f});
            OnnxTensor b = OnnxTensor.createTensor(env, new float[] {3.0f});
            OnnxTensor c = OnnxTensor.createTensor(env, new float[] {5.0f});

            // Request all outputs, supply all inputs
            inputMap.put("a:0", a);
            inputMap.put("b:0", b);
            inputMap.put("c:0", c);
            requestedOutputs.add("ab:0");
            requestedOutputs.add("bc:0");
            requestedOutputs.add("abc:0");
            try (Result r = session.run(inputMap, requestedOutputs)) {
                assertEquals(3, r.size());
                float abVal = unwrapFunc.apply(r, "ab:0");
                assertEquals(6.0f, abVal, 1e-10);
                float bcVal = unwrapFunc.apply(r, "bc:0");
                assertEquals(15.0f, bcVal, 1e-10);
                float abcVal = unwrapFunc.apply(r, "abc:0");
                assertEquals(21.0f, abcVal, 1e-10);
            }

            // Don't specify an output, expect all of them returned.
            try (Result r = session.run(inputMap)) {
                assertEquals(3, r.size());
                float abVal = unwrapFunc.apply(r, "ab:0");
                assertEquals(6.0f, abVal, 1e-10);
                float bcVal = unwrapFunc.apply(r, "bc:0");
                assertEquals(15.0f, bcVal, 1e-10);
                float abcVal = unwrapFunc.apply(r, "abc:0");
                assertEquals(21.0f, abcVal, 1e-10);
            }

            inputMap.clear();
            requestedOutputs.clear();

            // Request single output ab, supply all inputs
            inputMap.put("a:0", a);
            inputMap.put("b:0", b);
            inputMap.put("c:0", c);
            requestedOutputs.add("ab:0");
            try (Result r = session.run(inputMap, requestedOutputs)) {
                assertEquals(1, r.size());
                float abVal = unwrapFunc.apply(r, "ab:0");
                assertEquals(6.0f, abVal, 1e-10);
            }
            inputMap.clear();
            requestedOutputs.clear();

            // Request single output abc, supply all inputs
            inputMap.put("a:0", a);
            inputMap.put("b:0", b);
            inputMap.put("c:0", c);
            requestedOutputs.add("abc:0");
            try (Result r = session.run(inputMap, requestedOutputs)) {
                assertEquals(1, r.size());
                float abcVal = unwrapFunc.apply(r, "abc:0");
                assertEquals(21.0f, abcVal, 1e-10);
            }
            inputMap.clear();
            requestedOutputs.clear();

      /* The native library does all the computations, rather than the requested subset.
       * Leaving these tests commented out until it's fixed.
      // Request single output ab, supply required inputs
      inputMap.put("a:0",a);
      inputMap.put("b:0",b);
      requestedOutputs.add("ab:0");
      try (Result r = session.run(inputMap,requestedOutputs)) {
          assertEquals(1,r.size());
          float abVal = unwrapFunc.apply(r,"ab:0");
          assertEquals(6.0f,abVal,1e-10);
      }
      inputMap.clear();
      requestedOutputs.clear();

      // Request single output bc, supply required inputs
      inputMap.put("b:0",b);
      inputMap.put("c:0",c);
      requestedOutputs.add("bc:0");
      try (Result r = session.run(inputMap,requestedOutputs)) {
          assertEquals(1,r.size());
          float bcVal = unwrapFunc.apply(r,"bc:0");
          assertEquals(15.0f,bcVal,1e-10);
      }
      inputMap.clear();
      requestedOutputs.clear();
      */

            // Request output but don't supply the inputs
            inputMap.put("c:0", c);
            requestedOutputs.add("ab:0");
            try (Result r = session.run(inputMap, requestedOutputs)) {
                fail("Expected to throw OrtException due to incorrect inputs");
            } catch (OrtException e) {
                // System.out.println(e.getMessage());
                // pass
            }
            inputMap.clear();
            requestedOutputs.clear();

            // Request output but don't supply all the inputs
            inputMap.put("b:0", b);
            requestedOutputs.add("ab:0");
            try (Result r = session.run(inputMap, requestedOutputs)) {
                fail("Expected to throw OrtException due to incorrect inputs");
            } catch (OrtException e) {
                // System.out.println(e.getMessage());
                // pass
            }
        }
    }



}
