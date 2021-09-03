package com.bosuyun.ai.serving.factory;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.bosuyun.ai.serving.utils.ResourceUtils;
import org.junit.jupiter.api.Test;

class OrtSessionFactoryTest {

    @Test
    void test() {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession session;
        String modelPath = ResourceUtils.getPath("mnist-8.onnx");
        try {
            session = env.createSession(modelPath, new OrtSession.SessionOptions());
            System.out.println(session.getInputInfo());

        } catch (OrtException e) {
            e.printStackTrace();
        }

//        OnnxTensor t1, t2;
//        var inputs = Map.of("name1", t1, "name2", t2);
//        try (var results = session.run(inputs)) {
//            // manipulate the results
//        }
    }

}