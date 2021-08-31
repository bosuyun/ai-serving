package com.bosuyun.ai.serving.factory;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.util.Objects;

/**
 * Created by liuyuancheng on 2021/8/28  <br/>
 *
 * @author liuyuancheng
 */
public class OrtSessionFactory {

    private OrtSessionFactory() {

    }

    public static OrtSession getCpuSession(String modelPath, OrtSession.SessionOptions sessionOptions) {
        if (Objects.isNull(sessionOptions)) {
            sessionOptions = new OrtSession.SessionOptions();
        }
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        try {
            return env.createSession(modelPath, sessionOptions);
        } catch (OrtException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static OrtSession getGpuSession(String modelPath, OrtSession.SessionOptions sessionOptions, int deviceId) {
        if (Objects.isNull(sessionOptions)) {
            sessionOptions = new OrtSession.SessionOptions();
        }
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        try {
            sessionOptions.addCUDA(deviceId);
            return env.createSession(modelPath, sessionOptions);
        } catch (OrtException e) {
            e.printStackTrace();
        }
        return null;
    }
}
