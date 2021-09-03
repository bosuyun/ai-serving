package com.bosuyun.ai.serving.utils;

import com.google.common.io.Resources;
import org.apache.commons.lang3.StringUtils;

/**
 * Created by liuyuancheng on 2021/9/3  <br/>
 *
 * @author liuyuancheng
 */
public class ResourceUtils {

    public static String getPath(final String filename) {
        return Resources.getResource(ResourceUtils.class, StringUtils.prependIfMissing(filename, "/")).getPath();
    }

    public static void main(String[] args) {
        System.out.println(getPath("partial-inputs-test-2.onnx"));
    }

}
