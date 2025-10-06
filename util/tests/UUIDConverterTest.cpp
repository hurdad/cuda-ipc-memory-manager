#include <gtest/gtest.h>

#include "UUIDConverter.hpp"

using util::UUIDConverter;

TEST(UUIDConverterTest, ToFlatBufferUUID_CorrectConversion) {
    boost::uuids::uuid boostUuid = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF
    };

    fbs::cuda::ipc::api::UUID flatUuid = UUIDConverter::toFlatBufferUUID(boostUuid);

    uint64_t expected_msb, expected_lsb;
    std::memcpy(&expected_msb, boostUuid.data, sizeof(uint64_t));
    std::memcpy(&expected_lsb, boostUuid.data + sizeof(uint64_t), sizeof(uint64_t));

    EXPECT_EQ(flatUuid.msb(), expected_msb);
    EXPECT_EQ(flatUuid.lsb(), expected_lsb);
}

TEST(UUIDConverterTest, ToBoostUUID_CorrectConversion) {
    uint64_t msb = 0x0011223344556677;
    uint64_t lsb = 0x8899AABBCCDDEEFF;

    fbs::cuda::ipc::api::UUID flatUuid(msb, lsb);
    boost::uuids::uuid boostUuid = UUIDConverter::toBoostUUID(flatUuid);

    uint64_t result_msb, result_lsb;
    std::memcpy(&result_msb, boostUuid.data, sizeof(uint64_t));
    std::memcpy(&result_lsb, boostUuid.data + sizeof(uint64_t), sizeof(uint64_t));

    EXPECT_EQ(result_msb, msb);
    EXPECT_EQ(result_lsb, lsb);
}

TEST(UUIDConverterTest, RoundTripConversion) {
    boost::uuids::uuid original = {
        0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80,
        0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0, 0x00
    };

    fbs::cuda::ipc::api::UUID flatUuid = UUIDConverter::toFlatBufferUUID(original);
    boost::uuids::uuid convertedBack = UUIDConverter::toBoostUUID(flatUuid);

    EXPECT_EQ(original, convertedBack);
}

TEST(UUIDConverterTest, AllZeroUUID) {
    boost::uuids::uuid zeroUuid = {};
    fbs::cuda::ipc::api::UUID flatUuid = UUIDConverter::toFlatBufferUUID(zeroUuid);

    EXPECT_EQ(flatUuid.msb(), 0u);
    EXPECT_EQ(flatUuid.lsb(), 0u);

    boost::uuids::uuid roundTrip = UUIDConverter::toBoostUUID(flatUuid);
    EXPECT_EQ(zeroUuid, roundTrip);
}

TEST(UUIDConverterTest, AllFFUUID) {
    boost::uuids::uuid allFFUuid;
    std::memset(allFFUuid.data, 0xFF, 16);

    fbs::cuda::ipc::api::UUID flatUuid = UUIDConverter::toFlatBufferUUID(allFFUuid);

    EXPECT_EQ(flatUuid.msb(), UINT64_MAX);
    EXPECT_EQ(flatUuid.lsb(), UINT64_MAX);

    boost::uuids::uuid roundTrip = UUIDConverter::toBoostUUID(flatUuid);
    EXPECT_EQ(allFFUuid, roundTrip);
}
