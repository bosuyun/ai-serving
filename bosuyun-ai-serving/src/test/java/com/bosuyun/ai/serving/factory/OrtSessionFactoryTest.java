package com.bosuyun.ai.serving.factory;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.junit.jupiter.api.Test;

class OrtSessionFactoryTest {

    @Test
    void test() {
        System.out.println(System.getProperty("user.dir"));
        System.out.println(System.getProperty("os.name", "generic"));
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        try {
            env.createSession("mnist-8.onnx", new OrtSession.SessionOptions());
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

    @Test
    void getSession() throws OrtException {
        OrtSession ortSession = OrtSessionFactory.getCpuSession("mnist-8.onnx",
                new OrtSession.SessionOptions());
        System.out.println(ortSession.getInputInfo());
    }
}